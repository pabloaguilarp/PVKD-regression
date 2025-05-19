import numpy as np
import glob

def read_points(path):
  scan = np.fromfile(path, dtype=np.float32)
  scan = scan.reshape((-1, 4))  # [x,y,z,intensity]
  return scan

def read_label(path):
  label = np.fromfile(path, dtype=np.uint32)
  label = label.reshape((-1))
  return label

if __name__ == '__main__':
    files = glob.glob('./*/velodyne/*.bin')

    for f_path in files:
      scan = read_points(f_path)

      label_path = f_path.replace('velodyne', 'labels').replace('bin', 'label')
      labels = read_label(label_path)

      assert scan.shape[0] == labels.shape[0]