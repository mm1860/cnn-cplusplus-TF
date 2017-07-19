#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;
using namespace chrono;

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

string train_labels_file = "dataset_train.txt";
string test_labels_file = "dataset_test.txt";

#define VERBOSE true
#define VERBOSE_TSHD 10
#define NUM_ITERATIONS 100
#define BATCH_SIZE 25
#define IMAGE_HEIGHT 224
#define IMAGE_WIDTH 224
#define TOTAL_NUM_EXAMPLES_TRAIN 7316
#define TOTAL_NUM_EXAMPLES_TEST 1829
#define NUM_CHANNELS 3

void read_labels_file (string fileName, vector<pair<int, string>>& labelsFiles) {

	ifstream inFile(fileName);
	if (inFile.is_open()) {
		int label;
		string path;
		char c;
		while (inFile >> label >> c >> path && c == ',')
			labelsFiles.push_back (pair<int, string> (label, path));

	} else{
		cout << "[ERROR] Cannot open " << fileName << endl;
		exit (0);
	}

	inFile.close();

}

void print_labels_files_vector (vector<pair<int, string>>& labelsFiles) {
	int counter = 0;
	for (auto const& value: labelsFiles) {
		cout << "[VERBOSE] label=" << get<0>(value) << ", path=" << get<1>(value) << endl;
		if (counter == VERBOSE_TSHD)
			break;
		counter++;
	}
}

static Status ReadEntireFile (tensorflow::Env* env,
                              string& filename,
                              tensorflow::Tensor* output) {

  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR (env->GetFileSize (filename, &file_size));

  string contents;
  contents.resize (file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_CHECK_OK (env->NewRandomAccessFile (filename, &file));

  tensorflow::StringPiece data;
  TF_CHECK_OK (file->Read (0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss ("Truncated read of '", filename,
                                         "' expected ", file_size, " got ",
                                         data.size());
  }

  output->scalar<string>()() = data.ToString();
  return Status::OK();

}

int main (int argc, char* argv[]) {

	cout << "[INFO] start intitialize session" << endl;
	tensorflow::Session *session;

	TF_CHECK_OK (tensorflow::NewSession (tensorflow::SessionOptions(), &session));
	cout << "[INFO] Initialization Successful" << endl;

	tensorflow::GraphDef graph_def;
	TF_CHECK_OK (ReadBinaryProto (tensorflow::Env::Default(), "./models/graph.pb", &graph_def));
	TF_CHECK_OK (session->Create (graph_def));
	cout << "[INFO] Model has been loaded Successfully!" << endl;

	TF_CHECK_OK (session->Run({}, {}, {"init_all_vars_op"}, nullptr));
	cout << "[INFO] Preparing input data..." << endl;

	// Read dataset train label+imagepath file, so at each iteration a batch of image paths
	// with their corresponding labels could be chosen
	vector<pair<int, string>> trainLabelsFiles;
	vector<pair<int, string>> testLabelsFiles;
 	read_labels_file (train_labels_file, trainLabelsFiles);
 	read_labels_file (test_labels_file, testLabelsFiles);
 	if (VERBOSE) {
 		cout << "[VERBOSE] Print contents of trainLabelsFiles:" << endl;
 		print_labels_files_vector (trainLabelsFiles);
 		cout << endl;
 		cout << "[VERBOSE] Print contents of testLabelsFiles" << endl;
 		print_labels_files_vector (testLabelsFiles);
 		cout << endl;
 	}

  int startIndex = 0;

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    // Get the image from disk as float array of numbers and resized to the specifications
    // the main graph expects.
    vector<Tensor> resized_tensors_full;

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    vector<string> filenames;
    vector<float> labels;
    for (int j = 0; j < BATCH_SIZE; j++) {
      if (startIndex + j >= TOTAL_NUM_EXAMPLES_TRAIN)
        startIndex = 0;
      pair<int, string> currentPair = trainLabelsFiles[startIndex + j];
      filenames.push_back (currentPair.second);
      labels.push_back (currentPair.first);
    }

    if (startIndex + BATCH_SIZE >= TOTAL_NUM_EXAMPLES_TRAIN)
      startIndex = 0;
    else
      startIndex += BATCH_SIZE;

    // Preparing a vector of tensors
    vector<tensorflow::Tensor> inputsv;
    for (int j = 0; j < BATCH_SIZE; j++) {
      tensorflow::Tensor outTensors(tensorflow::DT_STRING, tensorflow::TensorShape());
      TF_CHECK_OK (ReadEntireFile (tensorflow::Env::Default(), filenames.at(j), &outTensors));

      inputsv.push_back (outTensors);
    }

    // Use a placeholder to read input datas
    int index = 0;
    vector<Tensor> resized_tensors;
    auto file_reader = Placeholder (root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    // Now we imagine it should be jpg, so we gonna decode it as jpg
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    image_reader = DecodeJpeg (root.WithOpName("jpeg_reader"), file_reader, DecodeJpeg::Channels(wanted_channels));

    auto float_caster = Cast (root.WithOpName ("float_caster"), image_reader, tensorflow::DT_FLOAT);
    auto dims_expander = ExpandDims (root.WithOpName ("dims_expander"), float_caster, 0);
    auto resized = ResizeBilinear (root.WithOpName("size"), dims_expander, Const (root, {IMAGE_HEIGHT, IMAGE_WIDTH}));
    tensorflow::GraphDef graph;
    TF_CHECK_OK (root.ToGraphDef(&graph));

    unique_ptr<tensorflow::Session> img_session (tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_CHECK_OK (img_session->Create(graph));
    for (tensorflow::Tensor inputTensor : inputsv) {
      vector<Tensor> resized_tensors;
      vector<pair<string, tensorflow::Tensor>> inputs = {{"input", inputTensor},};
      TF_CHECK_OK (img_session->Run({inputs}, {"size"}, {}, &resized_tensors));
      resized_tensors_full.push_back (resized_tensors.at(index));
    }

    // merge all tensors into one
    tensorflow::Tensor input_combined (tensorflow::DT_FLOAT, tensorflow::TensorShape({BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS}));
    tensorflow::Tensor label_combined (tensorflow::DT_FLOAT, tensorflow::TensorShape({BATCH_SIZE}));
    auto input_combined_flat = input_combined.flat<float>().data();
    for (int j = 0; j < BATCH_SIZE; j++) {
      auto resized_tensor_flat = resized_tensors_full.at(j).flat<float>().data();
      copy_n (resized_tensor_flat, IMAGE_WIDTH*IMAGE_HEIGHT*NUM_CHANNELS, input_combined_flat);
      input_combined_flat += IMAGE_WIDTH*IMAGE_HEIGHT*NUM_CHANNELS;
    }
    auto label_combined_flat = label_combined.flat<float>().data();
    copy_n (labels.begin(), BATCH_SIZE, label_combined_flat);


    vector<tensorflow::Tensor> costOutput;
    // Do one Step training here
    // First calculate loss value 
    TF_CHECK_OK (session->Run ({{"input", input_combined}, {"label", label_combined}}, {}, {"train"}, nullptr));
    TF_CHECK_OK (session->Run ({{"input", input_combined}, {"label", label_combined}}, {"loss"}, {}, &costOutput));
    float cost = costOutput[0].scalar<float>()(0);
    cout << "Cost value is: " << cost << endl;
    costOutput.clear();

  }

}
