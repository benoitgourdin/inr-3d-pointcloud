import os

if __name__ == '__main__':
    data_files = os.listdir('/home/mil/gourdin/inr_3d_data/data/medshapenet_vertebra')
    file = open('/inr-implicit-shape-reconstruction-mesh/casename_files/medshapenet/test_cases.txt', 'w')
    file.close()
    file = open('/inr-implicit-shape-reconstruction-mesh/casename_files/medshapenet/train_cases.txt', 'w')
    file.close()
    names = []
    for data in data_files:
        names.append(data.split('.')[0])
    with open('/inr-implicit-shape-reconstruction-mesh/casename_files/medshapenet/test_cases.txt', 'w') as f:
        for line in names:
            f.write(f"{line}\n")
    with open('/inr-implicit-shape-reconstruction-mesh/casename_files/medshapenet/train_cases.txt', 'w') as f:
        for line in names:
            f.write(f"{line}\n")
