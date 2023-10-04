'''
for transform the image
e.g. (1024x512->512x512)
没事的  ， 反正 clip 也会 先 pre_process()
'''
from abc import abstractmethod
import torchvision.transforms as transforms


class TransformsConfig(object):

	def __init__(self):
		pass

	@abstractmethod
	def get_transforms(self):
		pass

class SHHQTransforms(TransformsConfig):
	def __init__(self):
		super(SHHQTransforms, self).__init__()
	def get_transforms(self):
		transforms_dict = {
			'train':transforms.Compose([
				transforms.Resize((512, 512)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			# 那应该 from_image 与 to_image应该是一个；
			'test': transforms.Compose([
				transforms.Resize((512, 512)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((512, 512)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict