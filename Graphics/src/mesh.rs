use tobj;

// internal helper
fn generate_color_vec(color: [f32; 4], num: usize) -> Vec<f32> 
{
    color.iter().cloned().cycle().take(num*4).collect()
}

// Mesh

pub struct Mesh {
    pub vertices    : Vec<f32>,
    pub normals     : Vec<f32>,
    pub colors      : Vec<f32>,
    pub indices     : Vec<u32>,
    pub index_count : i32,
}

impl Mesh 
{
    pub fn from(mesh: tobj::Mesh, color: [f32; 4]) -> Self 
    {
        let num_verts = mesh.positions.len() / 3;
        let index_count = mesh.indices.len() as i32;
        Mesh 
        {
            vertices: mesh.positions,
            normals: mesh.normals,
            indices: mesh.indices,
            colors: generate_color_vec(color, num_verts),
            index_count,
        }
    }
}

// Body

pub struct Body;
impl Body 
{
    pub fn load(path: &str, color: [f32; 4]) -> Mesh 
    {
        let before = std::time::Instant::now();
        let (models, _materials)
            = tobj::load_obj(path,
                &tobj::LoadOptions
                {
                    triangulate: true,
                    single_index: true,
                    ..Default::default()
                }
            ).expect("Failed to load body model");
        let after = std::time::Instant::now();

        if models.len() > 1 || models.len() == 0 
        {
            panic!("Please use a model with a single mesh!")
        }

        let body = models[0].to_owned();
        Mesh::from(body.mesh, color)
    }
}


