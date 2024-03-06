Image = imread("C:/Users/Shahnawaz/OneDrive/Desktop/shahnawaz/passport size photo.jpg");
[count,x] = imhist(Image(:,:,1)); % select one of 3 channels