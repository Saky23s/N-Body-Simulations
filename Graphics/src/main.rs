// Comment these following global attributes to see most warnings of "low" interest:
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]
#![allow(unused_variables)]

extern crate nalgebra_glm as glm;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{mem, os::raw::c_void, ptr};
use std::time::Duration;

mod shader;
mod util;
mod mesh;
mod scene_graph;

use glutin::event::{
    DeviceEvent,
    ElementState::{Pressed, Released},
    Event, KeyboardInput,
    VirtualKeyCode::{self, *},
    WindowEvent,
};
use glutin::event_loop::ControlFlow;
use scene_graph::SceneNode;
use shader::Shader;

// initial window size
const INITIAL_SCREEN_W: u32 = 1920;
const INITIAL_SCREEN_H: u32 = 1080;

// Helper functions to make interacting with OpenGL a little bit prettier. 

// Get the size of an arbitrary array of numbers measured in bytes
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

// Get the OpenGL-compatible pointer to an arbitrary array of numbers
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

// Get the size of the given type in bytes
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

// Get an offset in bytes for n units of type T, represented as a relative pointer
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}


//Generate VAO 
unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>, colors: &Vec<f32>, normals: &Vec<f32>) -> u32 {
    // Create VAO and bind it
    let mut vao: u32 = 0;
    gl::GenVertexArrays(1, &mut vao as *mut u32);
    gl::BindVertexArray(vao);

    // Create position VBO and bind it
    let mut vbo: u32 = 0;
    gl::GenBuffers(1, &mut vbo as *mut u32);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
    gl::BufferData(gl::ARRAY_BUFFER, byte_size_of_array(vertices), pointer_to_array(vertices), gl::STATIC_DRAW);

    // Configure position VAP and enable it
    let index = 0;
    gl::VertexAttribPointer(index, 3, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
    gl::EnableVertexAttribArray(index);

    // Create color VBO
    let mut col_vbo: u32 = 0;
    gl::GenBuffers(1, &mut col_vbo as *mut u32);
    gl::BindBuffer(gl::ARRAY_BUFFER, col_vbo);
    gl::BufferData(gl::ARRAY_BUFFER, byte_size_of_array(colors), pointer_to_array(colors), gl::STATIC_DRAW);

    // Configure color VAP and enable it
    let col_index: u32 = 1;
    gl::VertexAttribPointer(col_index, 4, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
    gl::EnableVertexAttribArray(col_index);

    // Create normals VBO and bind it
    let mut normal_vbo: u32 = 0;
    gl::GenBuffers(1, &mut normal_vbo as *mut u32);
    gl::BindBuffer(gl::ARRAY_BUFFER, normal_vbo);
    gl::BufferData(gl::ARRAY_BUFFER, byte_size_of_array(normals), pointer_to_array(normals), gl::STATIC_DRAW);

    // Configure normals VAP and enable it
    let normal_index: u32 = 2;
    gl::VertexAttribPointer(normal_index, 3, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
    gl::EnableVertexAttribArray(normal_index);

    // Generate IBO and bind it
    let mut ibo: u32 = 0;
    gl::GenBuffers(1, &mut ibo as *mut u32);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ibo);
    gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, byte_size_of_array(indices), pointer_to_array(indices), gl::STATIC_DRAW);

    // Return ID of VAO
    return vao;

}

//Apply all trasformations to draw the root scene and to all of its children
unsafe fn draw_root(node: &scene_graph::SceneNode, shaders: &Shader, view_projection_matrix: &glm::Mat4, transformation_so_far: &glm::Mat4) 
{
    // Perform any logic needed before drawing the node
    let mut trasformation :glm::Mat4 = *transformation_so_far;
    
    // Check if node is drawable, if so: set uniforms, bind VAO and draw VAO
    if node.vao_id != 0 && node.index_count > 0
    {   
        let mut view :glm::Mat4  = *view_projection_matrix;
        let mut node_translation: glm::Mat4 = glm::translation(&glm::vec3(node.position[0], node.position[1], node.position[2]));
        node_translation = glm::scale(&node_translation,&node.scale); // Scale first, Translate second


        let mut reference_point_traslation: glm::Mat4 = glm::translation(&glm::vec3(node.reference_point[0], node.reference_point[1], node.reference_point[2]));
        let mut reference_point_traslation_inverse: glm::Mat4 = glm::translation(&glm::vec3(-node.reference_point[0], -node.reference_point[1], -node.reference_point[2]));
        let mut rotation_x: glm::Mat4 = glm::rotation(node.rotation[0], &glm::vec3(1.0, 0.0, 0.0));
        let mut rotation_y: glm::Mat4 = glm::rotation(node.rotation[1], &glm::vec3(0.0, 1.0, 0.0));
        let mut rotation_z: glm::Mat4 = glm::rotation(node.rotation[2], &glm::vec3(0.0, 0.0, 1.0));
        let mut final_camera_rotation: glm::Mat4 = reference_point_traslation * rotation_x * rotation_y * rotation_z * reference_point_traslation_inverse;

        
        trasformation  = trasformation  * node_translation * final_camera_rotation;
        view = view * trasformation;
        
        // Normals rotation
        let mut normal_transform: glm::Mat3 = glm::identity();
        normal_transform[(0, 0)] = trasformation[(0, 0)];
        normal_transform[(0, 1)] = trasformation[(0, 1)];
        normal_transform[(0, 2)] = trasformation[(0, 2)];
        normal_transform[(1, 0)] = trasformation[(1, 0)];
        normal_transform[(1, 1)] = trasformation[(1, 1)];
        normal_transform[(1, 2)] = trasformation[(1, 2)];
        normal_transform[(2, 0)] = trasformation[(2, 0)];
        normal_transform[(2, 1)] = trasformation[(2, 1)];
        normal_transform[(2, 2)] = trasformation[(2, 2)];

        gl::UniformMatrix3fv(shaders.get_uniform_location("normal_transform"), 1, gl::FALSE,normal_transform.as_ptr());
        gl::UniformMatrix4fv(shaders.get_uniform_location("transformation"), 1, gl::FALSE, view.as_ptr());
    }
    // Recurse
    for &child in &node.children 
    {   
        draw_sphere(&*child, shaders, view_projection_matrix, &trasformation);
    }
}

//Apply all trasformations to draw the root scene and to all of its children
unsafe fn draw_sphere(node: &scene_graph::SceneNode, shaders: &Shader, view_projection_matrix: &glm::Mat4, transformation_so_far: &glm::Mat4) 
{
    // Perform any logic needed before drawing the node
    let mut trasformation :glm::Mat4 = *transformation_so_far;
    
    // Check if node is drawable, if so: set uniforms, bind VAO and draw VAO
    if node.vao_id != 0 && node.index_count > 0
    {   
        let mut view :glm::Mat4  = *view_projection_matrix;
        let mut node_translation: glm::Mat4 = glm::translation(&glm::vec3(node.position[0], node.position[1], node.position[2]));
        node_translation = glm::scale(&node_translation,&node.scale); // Scale first, Translate second


        let mut reference_point_traslation: glm::Mat4 = glm::translation(&glm::vec3(node.reference_point[0], node.reference_point[1], node.reference_point[2]));
        let mut reference_point_traslation_inverse: glm::Mat4 = glm::translation(&glm::vec3(-node.reference_point[0], -node.reference_point[1], -node.reference_point[2]));
        let mut rotation_x: glm::Mat4 = glm::rotation(node.rotation[0], &glm::vec3(1.0, 0.0, 0.0));
        let mut rotation_y: glm::Mat4 = glm::rotation(node.rotation[1], &glm::vec3(0.0, 1.0, 0.0));
        let mut rotation_z: glm::Mat4 = glm::rotation(node.rotation[2], &glm::vec3(0.0, 0.0, 1.0));
        let mut final_camera_rotation: glm::Mat4 = reference_point_traslation * rotation_x * rotation_y * rotation_z * reference_point_traslation_inverse;

        
        trasformation  = trasformation  * node_translation * final_camera_rotation;
        view = view * trasformation;
        
        // Normals rotation
        let mut normal_transform: glm::Mat3 = glm::identity();
        normal_transform[(0, 0)] = trasformation[(0, 0)];
        normal_transform[(0, 1)] = trasformation[(0, 1)];
        normal_transform[(0, 2)] = trasformation[(0, 2)];
        normal_transform[(1, 0)] = trasformation[(1, 0)];
        normal_transform[(1, 1)] = trasformation[(1, 1)];
        normal_transform[(1, 2)] = trasformation[(1, 2)];
        normal_transform[(2, 0)] = trasformation[(2, 0)];
        normal_transform[(2, 1)] = trasformation[(2, 1)];
        normal_transform[(2, 2)] = trasformation[(2, 2)];

        gl::UniformMatrix3fv(shaders.get_uniform_location("normal_transform"), 1, gl::FALSE,normal_transform.as_ptr());
        gl::UniformMatrix4fv(shaders.get_uniform_location("transformation"), 1, gl::FALSE, view.as_ptr());
        
        //node.print();

        gl::BindVertexArray(node.vao_id);
        gl::DrawElements(gl::TRIANGLES, node.index_count, gl::UNSIGNED_INT, std::ptr::null());
    }
}


fn main() 
{
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("N-Body-Simulation")
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize::new(
            INITIAL_SCREEN_W,
            INITIAL_SCREEN_H,
        ));

    let cb = glutin::ContextBuilder::new().with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();
    
    // Uncomment these in the future for mouse control (Or not)
    // windowed_context.window().set_cursor_grab(true).expect("failed to grab cursor");
    // windowed_context.window().set_cursor_visible(false);

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Set up shared tuple for tracking changes to the window size
    let arc_window_size = Arc::new(Mutex::new((INITIAL_SCREEN_W, INITIAL_SCREEN_H, false)));
    // Make a reference of this tuple to send to the render thread
    let window_size = Arc::clone(&arc_window_size);
    
    //Flag to be able to check if the window should now close
    let should_close = Arc::new(Mutex::new(false));
    // Clone a reference to the shared flag for use outside the event loop
    let should_close_clone = should_close.clone();
    
    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || 
        {
        // Acquire the OpenGL Context and load the function pointers.
        // This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe 
        {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        let mut window_aspect_ratio = INITIAL_SCREEN_W as f32 / INITIAL_SCREEN_H as f32;

        // Set up openGL
        unsafe 
        {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            //gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            //gl::BlendFunc(gl::ONE, gl::ONE);
            gl::BlendFuncSeparate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA, gl::ONE, gl::ONE);
            //gl::BlendEquation(gl::FUNC_ADD);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!(
                "{}: {}",
                util::get_gl_string(gl::VENDOR),
                util::get_gl_string(gl::RENDERER)
            );
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!(
                "GLSL\t: {}",
                util::get_gl_string(gl::SHADING_LANGUAGE_VERSION)
            );
        }

        //Set yp some different colors for the bodies
        let colors = [
            [1.0,0.8274509803921568,0.00392156862745098, 1.0], //Yellow
            [0.36470588235294116, 0.6784313725490196, 0.9215686274509803, 1.0], //Blue
            [0.7607843137254902,0.23137254901960785,0.12941176470588237, 1.0]
            ];
        let n_colors = colors.len();

        //Create an array of meshs of different colors
        let mut bodies: Vec<mesh::Mesh> = Vec::new();
        
        // Load the bodies with diferent colors
        for color in colors
        {
            let body = mesh::Body::load("resources/sphere.obj", color);
            bodies.push(body);
        }
        
        // Transformation matrixes
        let mut camera_rotation: Vec<f32> = vec![-0.0000010281801,-0.041666705];
        let mut camera_translation: Vec<f32> = vec![0.008020077,0.0,4.4239545];
        
        //Set up trasformation variable
        let mut transformation;

        let mut perspective_projection: glm::Mat4 =
            glm::perspective(
                window_aspect_ratio, 
                PI/4.0, 
                1.0, 
                1000.0
            );
        
        //Set up array of vaos with different colors 
        let mut bodies_vaos: Vec<u32> = Vec::new();
        for body in &bodies
        {
            let body_vao = unsafe { create_vao(&body.vertices, &body.indices, &body.colors, &body.normals) };
            bodies_vaos.push(body_vao)
        }
        
        //Create the nodes
        //Root node
        let mut root_node = SceneNode::new();
                
        //Create space to store the bodies nodes
        let mut bodies_vector: Vec<scene_graph::Node> = Vec::new();
        
        let mut t = 1;
        //Read body files
        let path = "./data/starting_positions.bin";
        let mut n = 0;
        
        match util::read_starting_data_bin(path) 
        {
            Ok(records) => 
            {
                // Process the records here
                for record in records 
                {   
                    //Create
                    bodies_vector.push(SceneNode::from_vao(bodies_vaos[n % n_colors], bodies[n % n_colors].index_count));
                    bodies_vector[n].calculate_reference_point(&bodies[n % n_colors].vertices);

                    //Body is the child of the main node
                    root_node.add_child(&bodies_vector[n]);

                    //Put body in the correct place
                    bodies_vector[n].position[0] = record.x;
                    bodies_vector[n].position[1] = record.y;
                    bodies_vector[n].position[2] = record.z;
                    
                    //Scale the body 
                    bodies_vector[n].scale[0] = record.radius;
                    bodies_vector[n].scale[1] = record.radius;
                    bodies_vector[n].scale[2] = record.radius;

                    n += 1;
                }
            }
            Err(err) => 
            {
                eprintln!("Error: {}", err);
                panic!("Couldn't load bodies");
            }
        }        
        
        //Load shaders
        let shaders = unsafe 
        {
            shader::ShaderBuilder::new()
                .attach_file("./shaders/simple.vert")
                .attach_file("./shaders/simple.frag")
                .link()
        };

        //Activate shaders
        unsafe 
        {
            shaders.activate();
        }



        // The main rendering loop
        let first_frame_time = std::time::Instant::now();
        let mut prevous_frame_time = first_frame_time;
        loop 
        {

            //Delta time used for camera movements
            let delta_time = (1.0 / 24.0) as f32  * 10.0;
            let start = std::time::Instant::now();
            
            // Handle resize events
            if let Ok(mut new_size) = window_size.lock() 
            {
                if new_size.2
                 {
                    context.resize(glutin::dpi::PhysicalSize::new(new_size.0, new_size.1));
                    window_aspect_ratio = new_size.0 as f32 / new_size.1 as f32;
                    (*new_size).2 = false;
                    println!("Resized");
                    unsafe 
                    {
                        gl::Viewport(0, 0, new_size.0 as i32, new_size.1 as i32);
                        //Adjust perspective to new window aspect_ratio
                        perspective_projection = glm::perspective(
                                                        window_aspect_ratio, 
                                                        PI/4.0, 
                                                        1.0, 
                                                        1000.0
                                                    );
                    }
                }
            }

            // Handle keyboard input
            if let Ok(keys) = pressed_keys.lock() 
            {    
                for key in keys.iter() 
                {
                    match key 
                    {
                        // The `VirtualKeyCode` enum is defined here:
                        //    https://docs.rs/winit/0.25.0/winit/event/enum.VirtualKeyCode.html
                        //Look up, down, right, left
                        VirtualKeyCode::Up => { camera_rotation[1] -= delta_time / 10.0; }
                        VirtualKeyCode::Down => { camera_rotation[1] += delta_time / 10.0; }
                        VirtualKeyCode::Right => { camera_rotation[0] += delta_time / 10.0; }
                        VirtualKeyCode::Left => { camera_rotation[0] -= delta_time / 10.0; }
                        
                        //Move foward, backwards, left and right to the direction you are looking
                        VirtualKeyCode::W => 
                        {
                            camera_translation[2] -= delta_time * camera_rotation[0].cos();
                            camera_translation[0] += delta_time * camera_rotation[0].sin();
                        }
                        VirtualKeyCode::S => 
                        {
                            camera_translation[2] += delta_time * camera_rotation[0].cos();
                            camera_translation[0] -= delta_time * camera_rotation[0].sin();
                        }
                        VirtualKeyCode::D => 
                        {
                            camera_translation[2] += delta_time * camera_rotation[0].sin();
                            camera_translation[0] += delta_time * camera_rotation[0].cos();
                        }
                        VirtualKeyCode::A => {
                            camera_translation[2] -= delta_time * camera_rotation[0].sin();
                            camera_translation[0] -= delta_time * camera_rotation[0].cos();
                        }
                        //Move up and down.... direction you are looking seems to be missing...
                        VirtualKeyCode::Space => { camera_translation[1] += delta_time; }
                        VirtualKeyCode::LShift => { camera_translation[1] -= delta_time; }
                        // default handler:
                        _ => {}
                    }
                }
            }

            //Camera transforms here 
            transformation = glm::identity();

            let mut rotation_x: glm::Mat4 = glm::rotation(camera_rotation[0], &glm::vec3(0.0, 1.0, 0.0));
            let mut rotation_y: glm::Mat4 = glm::rotation(camera_rotation[1], &glm::vec3(1.0, 0.0, 0.0));
            let mut translation: glm::Mat4 = glm::translation(&glm::vec3(- camera_translation[0], - camera_translation[1], - camera_translation[2]));
            let mut final_camera_rotation: glm::Mat4 = rotation_y * rotation_x;

            //Apply camera trasformation
            transformation = perspective_projection * final_camera_rotation * translation * transformation;

            unsafe 
            {
                // Clear the color and depth buffers
                gl::ClearColor(0.035, 0.046, 0.078, 1.0); // night sky, full opacity
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                //Change the animation values
                //If the next data file exists
                t += 1;
                let path = format!("./data/{}.bin", t);
                //If it does not exist end simulation
                if util::file_exists(&path) == false
                {   
                    //Give signal to close
                    *should_close_clone.lock().unwrap() = true;     
                    return;          
                }
                //Read new positions of bodies to animate
                let mut i = 0;
                match util::read_running_data_bin(&path) 
                {
                    Ok(records) => 
                    {
                        // Process the records here
                        for record in records 
                        {
                            //Put body in the correct place
                            bodies_vector[i].position[0] = record.x;
                            bodies_vector[i].position[1] = record.y;
                            bodies_vector[i].position[2] = record.z;

                            i += 1;
            
                        }
                    }
                    Err(err) => 
                    {
                        eprintln!("Error: {}", err);
                        panic!("Couldn't load bodies");
                    }
                }  

                }
            let before_draw = std::time::Instant::now();
            // Draw
            unsafe { draw_root(&root_node, &shaders, &transformation, &glm::identity()); }
            let after_draw = std::time::Instant::now();
            println!("{}, {}", (before_draw - start).as_secs_f32(), (after_draw - before_draw).as_secs_f32());
            // Compute time passed since the previous frame and since the start of the program
            let now = std::time::Instant::now();

            //If it has not been 1/24 of a second dont show next frame
            if now.duration_since(prevous_frame_time) < Duration::from_secs_f32(1.0 / 24.0)
            {   
                thread::sleep(Duration::from_secs_f32(1.0 / 24.0)  - now.duration_since(prevous_frame_time));
            }

            prevous_frame_time = now;
            // Display the new color buffer on the display
            context.swap_buffers().unwrap(); // we use "double buffering" to avoid artifacts

            //Debuggin stuff
            //println!("{},{}", camera_translation[0],camera_translation[2]);
            //println!("{},{}", camera_rotation[0],camera_rotation[1]);
        }
    });

    // == //
    // == // From here on down there are only internals.
    // == //

    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || 
    {
        if !render_thread.join().is_ok() 
        {
            if let Ok(mut health) = render_thread_watchdog.write() 
            {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events are initially handled
    el.run(move |event, _, control_flow| 
    {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() 
        {
            if *health == false 
            {
                *control_flow = ControlFlow::Exit;
            }
        }

        // Check if we ran out of data and close the simulation if thats the case
        let should_close = should_close.lock().unwrap();
        if *should_close 
        {   
            *control_flow = ControlFlow::Exit;
            return;
        }


        match event 
        {
            Event::WindowEvent 
            {
                event: WindowEvent::Resized(physical_size),
                ..
            } => {
                println!(
                    "New window size! width: {}, height: {}",
                    physical_size.width, physical_size.height
                );
                if let Ok(mut new_size) = arc_window_size.lock() {
                    *new_size = (physical_size.width, physical_size.height, true);
                }
            }
            
            Event::WindowEvent 
            {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent 
            {
                event:
                    WindowEvent::KeyboardInput 
                    {
                        input:
                            KeyboardInput 
                            {
                                state: key_state,
                                virtual_keycode: Some(keycode),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                if let Ok(mut keys) = arc_pressed_keys.lock() 
                {
                    match key_state 
                    {
                        Released => 
                        {
                            if keys.contains(&keycode) 
                            {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        }
                        Pressed => 
                        {
                            if !keys.contains(&keycode) 
                            {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle Escape and Q keys separately
                match keycode 
                {
                    Escape => 
                    {
                        *control_flow = ControlFlow::Exit;
                    }
                    Q => 
                    {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                }
            }
            Event::DeviceEvent //Currently mouse movement is not used for anything, this event can be deleted
            {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => 
            {   
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() 
                {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            }
            
            _ => {}
        }
    });
}
