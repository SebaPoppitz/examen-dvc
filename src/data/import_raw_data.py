import requests  # For making HTTP requests to download files
import os        # For interacting with the operating system


# Function to download a single file from a given URL and save it in the specified directory
def import_raw_data(raw_data_relative_path, filename, bucket_folder_url):
    '''Download the specified file from bucket_folder_url and save it in raw_data_relative_path'''
    
    # Ensure the specified folder exists; create it if it doesn't
    os.makedirs(raw_data_relative_path, exist_ok=True)
    
    # Construct the full URL for the file to be downloaded
    input_file = os.path.join(bucket_folder_url, filename)
    # Construct the full local path where the file will be saved
    output_file = os.path.join(raw_data_relative_path, filename)
    
    print(f'Downloading {input_file} as {filename}')
    
    # Make a GET request to download the file
    response = requests.get(input_file)
    
    # If the request is successful (status code 200)
    if response.status_code == 200:
        # Save the response content as a file
        with open(output_file, "wb") as text_file:
            text_file.write(response.content)
        print(f'{filename} downloaded successfully.')
        print(text_file)
    else:
        # Print an error message if the request fails
        print(f'Error accessing the object {input_file}:', response.status_code)


# Main function that sets default values and calls import_raw_data
def main(raw_data_relative_path="../../data/raw_data", 
         filename="raw.csv",
         bucket_folder_url="https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/"):
    """Download a single file from an AWS S3 bucket and save it in ./data/raw"""
    
    # Call the import_raw_data function to download the file
    import_raw_data(raw_data_relative_path, filename, bucket_folder_url)
    

# Entry point of the script
if __name__ == '__main__':
    
    # Call the main function to start the script
    main()
