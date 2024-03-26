use std::ffi::CString;
use libc;
use std::error::Error;
use std::fs::File;
use serde::Deserialize;
use csv::ReaderBuilder;
use std::fs;
use csv::Reader;

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
    pub mass: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
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