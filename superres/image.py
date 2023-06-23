import time
import cv2
import os

model = "models/EDSR_x4.pb"

image = "examples/suspect.jpg"

modelName = model.split(os.path.sep)[-1].split("_")[0].lower()
modelScale = model.split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model)
sr.setModel(modelName, modelScale)

image = cv2.imread(image)
print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
# use the super resolution model to upscale the image, timing how
# long it takes
start = time.time()
upscaled = sr.upsample(image)
end = time.time()
print("[INFO] super resolution took {:.6f} seconds".format(
	end - start))
# show the spatial dimensions of the super resolution image
print("[INFO] w: {}, h: {}".format(upscaled.shape[1],
	upscaled.shape[0]))


start = time.time()
bicubic = cv2.resize(image, (upscaled.shape[1], upscaled.shape[0]),
	interpolation=cv2.INTER_CUBIC)
end = time.time()
print("[INFO] bicubic interpolation took {:.6f} seconds".format(
	end - start))

#cv2.imshow("Original", image)
#cv2.imshow("Bicubic", bicubic)
#cv2.imshow("Super Resolution", upscaled)
cv2.imwrite("results/bicupic.jpg",bicubic)
cv2.imwrite("results/upscaled.jpg",upscaled)
#cv2.waitKey(0)