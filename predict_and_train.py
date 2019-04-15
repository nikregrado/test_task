from data_reading import get_train_data, get_test_data, mask_to_rle, resize, np
from model_unet import make_model
import pandas as pd

epochs = 60

# get train_data
train_img, train_mask = get_train_data()

# get test_data
test_img, test_img_sizes = get_test_data()

# get u_net model
u_net = make_model()


print("\nTraining...")
u_net.fit(train_img, train_mask, batch_size=16, epochs=epochs)

print("Predicting")
# Predict on test data
test_mask = u_net.predict(test_img, verbose=1)


test_mask_upsampled = []
for i in range(len(test_mask)):
    test_mask_upsampled.append(resize(np.squeeze(test_mask[i]), (test_img_sizes[i][0], test_img_sizes[i][1]),
                                      mode='constant', preserve_range=True))


test_ids, rles = mask_to_rle(test_mask_upsampled)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('submission.csv', index=False)
