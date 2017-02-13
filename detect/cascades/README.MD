# Cascades description

**simple_haar_00.xml** - This classifier trained on images of cars with divided background. This approach allows to use
simple classifier and works fast. This one trained on 30 positive examples. It works well for small cars but not good
with buses.

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/Desktop/opencv/OpenCV/modules \
    -D BUILD_EXAMPLES=ON ..


cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/home/pi/Desktop/opencv/OpenCV/modules \
	-D BUILD_EXAMPLES=ON ..


cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/home/pi/Desktop/opencv/opencv_contrib-3.1.0/modules \
	-D BUILD_EXAMPLES=ON ..