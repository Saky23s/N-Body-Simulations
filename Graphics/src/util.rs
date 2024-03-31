use std::ffi::CString;
use libc;
use std::error::Error;
use std::fs::File;
use serde::Deserialize;
use csv::ReaderBuilder;
use std::fs;
use csv::Reader;
use std::io::{self, Read};

pub unsafe fn get_gl_string(name: gl::types::GLenum) -> String {
    std::ffi::CStr::from_ptr(gl::GetString(name) as *mut libc::c_char).to_string_lossy().to_string()
}

// Debug callback to panic upon enountering any OpenGL error
pub extern "system" fn debug_callback(
    source: u32, e_type: u32, id: u32,
    severity: u32, _length: i32,
    msg: *const libc::c_char, _data: *mut std::ffi::c_void
) {
    if e_type != gl::DEBUG_TYPE_ERROR { return }
    if severity == gl::DEBUG_SEVERITY_HIGH ||
       severity == gl::DEBUG_SEVERITY_MEDIUM ||
       severity == gl::DEBUG_SEVERITY_LOW
    {
        let severity_string = match severity {
            gl::DEBUG_SEVERITY_HIGH => "high",
            gl::DEBUG_SEVERITY_MEDIUM => "medium",
            gl::DEBUG_SEVERITY_LOW => "low",
            _ => "unknown",
        };
        unsafe {
            let string = CString::from_raw(msg as *mut libc::c_char);
            let error_message = String::from_utf8_lossy(string.as_bytes()).to_string();
            panic!("{}: Error of severity {} raised from {}: {}\n",
                id, severity_string, source, error_message);
        }
    }
}

//Struct to be able to easily read from file
#[derive(Debug, Deserialize)]
pub struct RunningBodyData
{
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

//Struct to be able to easily read from file
#[derive(Debug, Deserialize)]
pub struct StartingBodyData
{
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub radius: f32,
}

//Helper funtion to check if a file exists
pub fn file_exists(path: &str) -> bool 
{
    if let Ok(metadata) = fs::metadata(path) { metadata.is_file() } 
    else { false }
}

//Funtion to read the new positions of all the bodies from a csv file
pub fn read_running_data_csv(path: &str) -> Result<Vec<RunningBodyData>, Box<dyn Error>> 
{
    // Opens the CSV file
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);

    // Collect records into a vector
    let mut records = Vec::new();
    
    //Read all rows
    for result in reader.deserialize() 
    {   
        //Create a RunningBodyData with that row
        let record: (f32, f32, f32) = result?;
        let RunningBodyData = RunningBodyData 
        {
            x: record.0,
            y: record.1,
            z: record.2,
        };
        //Add to the vector
        records.push(RunningBodyData);
    }
    //Return vector
    Ok(records)
}

//Funtion to read the starting data of all the bodies from a csv file
pub fn read_starting_data_csv(path: &str) -> Result<Vec<StartingBodyData>, Box<dyn Error>> 
{
    // Opens the CSV file
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);

    // Retrieve and print header record
    let headers = reader.headers()?.clone();

    // Collect records into a vector
    let records: Result<Vec<StartingBodyData>, csv::Error> = reader.deserialize().collect();
    let records = records.map_err(|e| <csv::Error as Into<Box<dyn Error>>>::into(e))?;

    Ok(records)
}

//Funtion to read the new positions of all the bodies from a bin file
pub fn read_running_data_bin(path: &str) -> Result<Vec<RunningBodyData>, Box<dyn Error>> 
{
    // Opens the bin file
    let mut file = File::open(path)?;

    // Load it to memory
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Collect records into a vector
    let mut records = Vec::new();

    // Check if the buffer size is multiple of the size of a record (3 doubles)
    if buffer.len() % (3 * std::mem::size_of::<f64>()) != 0 
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid file size",
        ).into());
    }

    if cfg!(target_endian = "little")
    {
        for chunk in buffer.chunks_exact(3 * std::mem::size_of::<f64>()) 
        {
            let mut x_bytes = [0; 8];
            x_bytes.copy_from_slice(&chunk[0..8]);
            let x = f64::from_le_bytes(x_bytes) as f32; //Read as doubles save as f32. 

            let mut y_bytes = [0; 8];
            y_bytes.copy_from_slice(&chunk[8..16]);
            let y = f64::from_le_bytes(y_bytes) as f32;

            let mut z_bytes = [0; 8];
            z_bytes.copy_from_slice(&chunk[16..24]);
            let z = f64::from_le_bytes(z_bytes) as f32;

            records.push(RunningBodyData { x, y, z });
        }
    }
    else
    {
        for chunk in buffer.chunks_exact(3 * std::mem::size_of::<f64>()) 
        {
            let mut x_bytes = [0; 8];
            x_bytes.copy_from_slice(&chunk[0..8]);
            let x = f64::from_be_bytes(x_bytes) as f32; //Read as doubles save as f32. 

            let mut y_bytes = [0; 8];
            y_bytes.copy_from_slice(&chunk[8..16]);
            let y = f64::from_be_bytes(y_bytes) as f32;

            let mut z_bytes = [0; 8];
            z_bytes.copy_from_slice(&chunk[16..24]);
            let z = f64::from_be_bytes(z_bytes) as f32;

            records.push(RunningBodyData { x, y, z });
        }
    }
    
    Ok(records)
}

//Funtion to read the starting data of all the bodies from a bin file. 
pub fn read_starting_data_bin(path: &str) -> Result<Vec<StartingBodyData>, Box<dyn Error>> 
{   

    // Opens the bin file
    let mut file = File::open(path)?;

    // Load it to memory
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Collect records into a vector
    let mut records = Vec::new();

    // Check if the buffer size is multiple of the size of a record (4 doubles)
    if buffer.len() % (4 * std::mem::size_of::<f64>()) != 0 
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid file size",
        ).into());
    }

    //Binary in little endian
    if cfg!(target_endian = "little")
    {
        for chunk in buffer.chunks_exact(4 * std::mem::size_of::<f64>()) 
        {
            let mut x_bytes = [0; 8];
            x_bytes.copy_from_slice(&chunk[0..8]);
            let x = f64::from_le_bytes(x_bytes) as f32; //Read as doubles save as f32. 
    
            let mut y_bytes = [0; 8];
            y_bytes.copy_from_slice(&chunk[8..16]);
            let y = f64::from_le_bytes(y_bytes) as f32;
    
            let mut z_bytes = [0; 8];
            z_bytes.copy_from_slice(&chunk[16..24]);
            let z = f64::from_le_bytes(z_bytes) as f32;
    
            let mut r_bytes = [0; 8];
            r_bytes.copy_from_slice(&chunk[24..32]);
            let radius = f64::from_le_bytes(r_bytes) as f32;
    
            records.push(StartingBodyData { x, y, z, radius });
        }
    }
    //Binary in big endian
    else
    {   
        for chunk in buffer.chunks_exact(4 * std::mem::size_of::<f64>()) 
        {
            let mut x_bytes = [0; 8];
            x_bytes.copy_from_slice(&chunk[0..8]);
            let x = f64::from_be_bytes(x_bytes) as f32; //Read as doubles save as f32. 
    
            let mut y_bytes = [0; 8];
            y_bytes.copy_from_slice(&chunk[8..16]);
            let y = f64::from_be_bytes(y_bytes) as f32;
    
            let mut z_bytes = [0; 8];
            z_bytes.copy_from_slice(&chunk[16..24]);
            let z = f64::from_be_bytes(z_bytes) as f32;
    
            let mut r_bytes = [0; 8];
            r_bytes.copy_from_slice(&chunk[24..32]);
            let radius = f64::from_be_bytes(r_bytes) as f32;
    
            records.push(StartingBodyData { x, y, z, radius });
        }
    }
    Ok(records)
}