import tensorflow as tf
import os.path


class RoadSignDataset:
    """
    A class to handle road sign dataset loading, preprocessing, and splitting.
    """

    def __init__(self, datasetDir, imageWidth, imageHeight, batchSize):
        """
        Initialize RoadSignDataset with dataset directory, image dimensions, and batch size.

        Parameters:
        - datasetDir (str): The directory path containing the road sign images.
        - imageWidth (int): The width of the images in the dataset.
        - imageHeight (int): The height of the images in the dataset.
        - batchSize (int): Size of the batches of data.

        Returns:
        - None
        """
        self.__DatasetDir = datasetDir
        self.__ImageWidth = imageWidth
        self.__ImageHeight = imageHeight
        self.__BatchSize = batchSize
        self.__autotune = tf.data.AUTOTUNE

        self.__LearnData = self.__loadData()

    def __loadData(self):
        """
        Load and preprocess the road sign dataset from the specified directory.

        Returns:
        - tf.data.Dataset: A TensorFlow dataset object containing the preprocessed road sign images and their corresponding labels.
        """
        return tf.keras.utils.image_dataset_from_directory(
            directory=self.__DatasetDir,
            shuffle=True,
            batch_size=None,
            image_size=(self.__ImageHeight, self.__ImageWidth),
        )

    def getClasses(self):
        """
        Get the class names of the road sign dataset.

        This function returns a list of class names corresponding to the road sign images in the dataset.
        The class names are extracted from the directory structure of the dataset directory.

        Parameters:
        - None

        Returns:
        - list: A list of strings representing the class names of the road sign dataset.
        """
        return self.__LearnData.class_names

    def split(self, dataset, trainRatio, validRatio):
        """
        Split the road sign dataset into training, validation, and testing sets based on given ratios.

        This function takes a TensorFlow dataset object, calculates the sizes of the training, validation,
        and testing sets based on the provided ratios, and then extracts the corresponding datasets.

        Parameters:
        - dataset (tf.data.Dataset): A TensorFlow dataset object containing the road sign images and their corresponding labels.
        - trainRatio (float): The ratio of the dataset to be used for training.
        - validRatio (float): The ratio of the dataset to be used for validation.

        Returns:
        - tuple: A tuple containing three TensorFlow dataset objects for training, validation, and testing respectively.
        """
        learnDataLength = len(list(dataset))
        trainSize = int(learnDataLength * trainRatio)
        validSize = int(learnDataLength * validRatio)
        testSize = learnDataLength - trainSize - validSize

        trainData = dataset.take(trainSize)
        validData = dataset.skip(trainSize).take(validSize)
        testData = dataset.skip(trainSize + validSize).take(testSize)
        return (
            trainData.batch(self.__BatchSize),
            validData.batch(self.__BatchSize),
            testData.batch(self.__BatchSize),
        )

    def getData(self, trainRatio=0.7, validRatio=0.2):
        """
        Get the preprocessed and split road sign dataset for training, validation, and testing.

        This function splits the loaded road sign dataset into training, validation, and testing sets based on the given ratios.
        It applies data prefetching to improve performance.

        Parameters:
        - trainRatio (float, optional): The ratio of the dataset to be used for training. Default is 0.7.
        - validRatio (float, optional): The ratio of the dataset to be used for validation. Default is 0.2.

        Returns:
        - tuple: A tuple containing three TensorFlow dataset objects for training, validation, and testing respectively.

        Raises:
        - None
        """
        trainData, validData, testData = self.split(
            self.__LearnData, trainRatio, validRatio
        )

        # Do buffered prefetching to avoid i/o blocking
        trainData = trainData.prefetch(buffer_size=self.__autotune)
        testData = testData.prefetch(buffer_size=self.__autotune)
        validData = validData.prefetch(buffer_size=self.__autotune)

        return trainData, validData, testData

    def __createExample(self, image, label):
        """
        Create a TensorFlow Example protocol buffer from an image and its corresponding label.

        This function takes an image tensor and its corresponding label, converts the image to uint8 dtype,
        encodes it as JPEG, and creates a TensorFlow Example protocol buffer containing the encoded image
        and its label.

        Parameters:
        - image (tf.Tensor): A tensor representing the input image. The shape of the tensor should be [height, width, channels].
        - label (int): The corresponding label for the input image.

        Returns:
        - tf.train.Example: A TensorFlow Example protocol buffer containing the encoded image and its label.
        """
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image = tf.io.encode_jpeg(image)

        feature = {
            "images": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image.numpy()])
            ),
            "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def __parseRecord(self, example):
        """
        Parse a single TFRecord example and decode the image and label.

        This function takes a single TFRecord example as input, which contains serialized image data and its corresponding label.
        It decodes the image data from the TFRecord example, converts it to a float32 tensor, and returns the image and label.

        Parameters:
        - example (tf.train.Example): A single TFRecord example containing serialized image data and its corresponding label.

        Returns:
        - tuple: A tuple containing the decoded image (tf.Tensor) and its corresponding label (tf.Tensor).
        """
        feature_description = {
            "images": tf.io.FixedLenFeature([], tf.string),
            "labels": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["images"] = tf.image.convert_image_dtype(
            tf.io.decode_jpeg(example["images"], channels=3), dtype=tf.float32
        )

        return example["images"], example["labels"]

    def save2Records(self, path):
        """
        Save the road sign dataset to a TFRecord file.

        This function takes the road sign dataset, converts it into TensorFlow Example protocol buffer format,
        and writes it to a TFRecord file. If the specified file already exists, it prints a message indicating
        that the file already exists.

        Parameters:
        - path (str): The file path where the TFRecord file will be saved.

        Returns:
        - None

        Raises:
        - None
        """
        if not os.path.isfile(path):
            with tf.io.TFRecordWriter(path) as writer:
                for image, label in self.__LearnData:
                    tf_example = self.__createExample(image, label)
                    writer.write(tf_example.SerializeToString())
        else:
            print("File already exists")

    def getDataFromRecords(self, path, trainRatio=0.7, validRatio=0.2):
        """
        Load and preprocess road sign dataset from TFRecord file.

        This function reads a TFRecord file containing road sign images and their corresponding labels,
        and then splits the dataset into training, validation, and testing sets based on the given ratios.
        The function also applies data prefetching to improve performance.

        Parameters:
        - path (str): The file path of the TFRecord file.
        - trainRatio (float, optional): The ratio of the dataset to be used for training. Default is 0.7.
        - validRatio (float, optional): The ratio of the dataset to be used for validation. Default is 0.2.

        Returns:
        - tuple: A tuple containing three TensorFlow dataset objects for training, validation, and testing respectively.

        Raises:
        - OSError: If the specified file does not exist.
        """
        if not os.path.isfile(path):
            print("File does not exist")
            raise OSError("File does not exist")

        # Create a dataset from the TFRecord file.
        rawDataset = tf.data.TFRecordDataset(path)
        dataset = rawDataset.map(self.__parseRecord)

        trainData, validData, testData = self.split(dataset, trainRatio, validRatio)

        # Do buffered prefetching to avoid i/o blocking
        trainData = trainData.prefetch(buffer_size=self.__autotune)
        testData = testData.prefetch(buffer_size=self.__autotune)
        validData = validData.prefetch(buffer_size=self.__autotune)

        return trainData, validData, testData
