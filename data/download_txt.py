import os.path
import os
import requests


# download MedShapeNet
if __name__ == '__main__':
    input_list = '/.../vertebra_medshapenet.txt'
    output_dir = '/.../medshapenet_vertebra/'
    file = open(input_list, 'r')
    data = file.read()
    data_list = data.split("\n")
    file.close()
    for url in data_list:
        r = requests.get(url, allow_redirects=True)
        file_name = url.split('=')[-1]
        new_path = os.path.join(output_dir, file_name)
        open(new_path, 'wb').write(r.content)
        print("downloaded: " + str(file_name))
