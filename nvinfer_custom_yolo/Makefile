CXX:=g++
SRCS:= nvdsparsebbox_yolo.cpp
TARGET_LIB:= libnvdsinfer_custom_impl_Yolo.so

DEEPSTREAM_SDK_PATH:=/opt/nvidia/deepstream/deepstream

CFLAGS:= -Wall -std=c++11 -shared -fPIC
CFLAGS+= -I$(DEEPSTREAM_SDK_PATH)/sources/includes
CFLAGS+= -I/usr/local/cuda/include

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCS)
	$(CXX) -o $@ $^ $(CFLAGS)

clean:
	rm -rf $(TARGET_LIB)
