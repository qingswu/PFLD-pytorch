import numpy as np


from pfld.dataset.tardataset import TarDataset


class WLFWTarDatasets(TarDataset):
    def __init__(self, tar_data, annotation_list, transforms=None) -> None:
        super().__init__(archive=tar_data)
        self.lines = self.get_text_file("WLFW_data/" + annotation_list).splitlines()
        self.transforms = transforms

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index].strip().split()
        img = self.get_image("WLFW_data/" + line[0], pil=True)
        landmark = np.asarray(line[1:197], dtype=np.float32)
        attribute = np.asarray(line[197:203], dtype=np.int32)
        euler_angle = np.asarray(line[203:206], dtype=np.float32)
        if self.transforms:
            img = self.transforms(img)
        return img, landmark, attribute, euler_angle
