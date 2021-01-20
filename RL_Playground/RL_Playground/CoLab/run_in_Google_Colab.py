# Thanks to PyDrive library GoogleColab can now be used form here.
# See https://stackoverflow.com/questions/48376580/google-colab-how-to-read-data-from-my-google-drive
# Change name of this file ! ! ! 

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


if __name__ == '__main__':
    #Login to Google Drive and create drive object
    g_login = GoogleAuth()
    g_login.LocalWebserverAuth()
    drive = GoogleDrive(g_login)
    
    upload_filename = 'Connect4_MCTreeSearch_train.py'
    folder_id = '11EwmXPkI_0zY09Fq7krWPb4h0Nb9TQML'

    with open(upload_filename, "r") as upload_file:
        filename = os.path.basename(upload_file.name)
        file_list = drive.ListFile({'q':"'" + folder_id + "' in parents and trashed=False"}).GetList()
        for file in file_list:
            print(file['title'])
            if file['title'] == filename:
                file.Delete()
        file_drive = drive.CreateFile({'title': filename,
                                       'parents': ['1ymVP8RygFFfuRKebrgHhanEH-0oJdwjT',
                                                   '1Vs8WSx2UuQe9iJ88-0NTqWQT32wvyQUy',
                                                   '18XYjUqW6o3UJlhRauJCXZj9NhDhPVfdE',
                                                   '10Tad7FwFEnmsbNU9CVFhbwCE4Cf7vfHw',
                                                   folder_id]})  
        file_drive.SetContentString(upload_file.read()) 
        file_drive.Upload()