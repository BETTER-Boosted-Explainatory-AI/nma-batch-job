import pandas as pd
import os

class EdgesDataframe:
    def __init__(self, model_filename, df_filename, edges_df=None):

        print(f"Dataframe file path: {df_filename}")
        
        self.model_filename = model_filename
        self.df_filename = df_filename
        self.edges_df = pd.DataFrame() if edges_df is None else edges_df

    def get_dataframe(self):
        return self.edges_df

    def save_dataframe(self):
        try:
            # Extract directory path from the file path
            directory = os.path.dirname(self.df_filename)
            
            # Create directories if they don't exist
            if directory:  # Only if there is a directory path
                os.makedirs(directory, exist_ok=True)
                print(f'Ensured directory exists: {directory}')
            
            # Check if file already exists
            if os.path.exists(self.df_filename):
                print(f'File already exists, skipping save: {self.df_filename}')
                return
            
            # Save the dataframe only if file doesn't exist
            self.edges_df.to_csv(self.df_filename, index=False)
            print(f'Edges dataframe has been saved: {self.df_filename}')
        
        except Exception as e:
            print(f'Error saving dataframe: {str(e)}')

    # def load_dataframe(self):
    #     if os.path.exists(self.df_filename):
    #         self.edges_df = pd.read_csv(self.df_filename)
    #         print(self.edges_df.head())
    #         print(f'Edges dataframe has been loaded: {self.df_filename}')
    #     else:
    #         print(f'File not found: {self.df_filename}')


    def load_dataframe(self):
        print(f"Attempting to load DataFrame from: {self.df_filename}")
        try:
            if os.path.exists(self.df_filename):
                print(f"File exists and has size: {os.path.getsize(self.df_filename)} bytes")
                self.edges_df = pd.read_csv(self.df_filename)
                print(f"DataFrame loaded with shape: {self.edges_df.shape}")
                print(f"DataFrame columns: {self.edges_df.columns.tolist()}")
                print("First few rows:")
                print(self.edges_df.head())
            else:
                print(f'File not found: {self.df_filename}')
        except Exception as e:
            print(f"Error loading DataFrame: {str(e)}")
            
    def get_image_probabilities_by_id(self, image_id):
        probabilities_df = self.edges_df[self.edges_df['image_id'] == image_id]
        return probabilities_df

    def get_dataframe_by_count(self):
        return self.edges_df.groupby('source')['target'].value_counts().reset_index(name='count')
