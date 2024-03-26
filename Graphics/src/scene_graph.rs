extern crate nalgebra_glm as glm;

use std::mem::ManuallyDrop;
use std::pin::Pin;

pub type Node = ManuallyDrop<Pin<Box<SceneNode>>>;

pub struct SceneNode {
    pub position        : glm::Vec3,   // Where I should be in relation to my parent
    pub rotation        : glm::Vec3,   // How I should be rotated, around the X, the Y and the Z axes
    pub scale           : glm::Vec3,   // How I should be scaled
    pub reference_point : glm::Vec3,   // The point I shall rotate and scale about

    pub vao_id      : u32,             // What I should draw
    pub index_count : i32,             // How much of it there is to draw

    pub children: Vec<*mut SceneNode>, // Those I command
}

impl SceneNode {

    pub fn new() -> Node {
        ManuallyDrop::new(Pin::new(Box::new(SceneNode {
            position        : glm::zero(),
            rotation        : glm::zero(),
            scale           : glm::vec3(1.0, 1.0, 1.0),
            reference_point : glm::zero(),
            vao_id          : 0,
            index_count     : -1,
            children        : vec![],
        })))
    }

    pub fn from_vao(vao_id: u32, index_count: i32) -> Node {
        ManuallyDrop::new(Pin::new(Box::new(SceneNode {
            position        : glm::zero(),
            rotation        : glm::zero(),
            scale           : glm::vec3(1.0, 1.0, 1.0),
            reference_point : glm::zero(),
            vao_id,
            index_count,
            children: vec![],
        })))
    }

    pub fn add_child(&mut self, child: &SceneNode) {
        self.children.push(child as *const SceneNode as *mut SceneNode)
    }

    pub fn calculate_reference_point(&mut self, vertices: &Vec<f32>)
    {   
        //Set the reference point at the geometric center of the object
        let mut sum = glm::vec3(0.0, 0.0, 0.0);
        let mut i : usize = 0;
        let mut counter = 0.0;
        let len = vertices.len();
        while i < len
        {   
            let mut point = glm::vec3(vertices[i], vertices[i + 1], vertices[i + 2]);
            sum[0] += point[0];
            sum[1] += point[1];
            sum[2] += point[2];
            i += 3;
            counter += 1.0;
        }
        self.reference_point = glm::vec3(
            (sum[0]) / counter,
            (sum[1]) / counter,
            (sum[2]) / counter,
        );
    }

    #[allow(dead_code)]
    pub fn get_child(& mut self, index: usize) -> & mut SceneNode {
        unsafe {
            &mut (*self.children[index])
        }
    }

    #[allow(dead_code)]
    pub fn get_n_children(&self) -> usize {
        self.children.len()
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        println!(
"SceneNode {{
    VAO:       {}
    Indices:   {}
    Children:  {}
    Position:  [{:.2}, {:.2}, {:.2}]
    Rotation:  [{:.2}, {:.2}, {:.2}]
    Reference: [{:.2}, {:.2}, {:.2}]
}}",
            self.vao_id,
            self.index_count,
            self.children.len(),
            self.position.x,
            self.position.y,
            self.position.z,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
            self.reference_point.x,
            self.reference_point.y,
            self.reference_point.z,
        );
    }

}


// You can also use square brackets to access the children of a SceneNode
use std::ops::{Index, IndexMut};
impl Index<usize> for SceneNode {
    type Output = SceneNode;
    fn index(&self, index: usize) -> &SceneNode {
        unsafe {
            & *(self.children[index] as *const SceneNode)
        }
    }
}
impl IndexMut<usize> for SceneNode {
    fn index_mut(&mut self, index: usize) -> &mut SceneNode {
        unsafe {
            &mut (*self.children[index])
        }
    }
}
