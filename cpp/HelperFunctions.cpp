
#include "Task_2.h"

// Function to read the csv file !!!

std::vector<std::vector<float>> readCSV(const std::string& filename) {

	std::vector<std::vector<float>> data;
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Error: Could not open file " << filename << std::endl;
		return data;
	}
	else {

		try {
			std::string line = "";
			while (getline(file, line)) {
				std::vector<float> row;
				std::istringstream lineStream(line);
				std::string cell;

				while (getline(lineStream, cell, ',')) {
					//row.push_back(cell);
					//std::stringstream values(cell);
						/*float value;
					cell >> value;*/
					row.push_back(std::stof(cell));
				}

				data.push_back(row);
			}
		}
		catch (const std::exception e) {
			std::cerr << "An error occured :" << e.what() << std::endl;
		}
		file.close();
		return data;
	}
}


// Function to do Train Test split !!!

std::tuple <torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TrainTestSplit(torch::Tensor X, torch::Tensor y) {

	int n_samples = X.size(0);
	int n_train = n_samples * 0.8;

	torch::manual_seed(42);
	torch::Tensor indices = torch::randperm(n_samples);
	torch::Tensor train_indices = indices.slice(0, 0, n_train);
	torch::Tensor test_indices = indices.slice(0, n_train);


	torch::Tensor X_train = X.index({ train_indices });
	torch::Tensor y_train = y.index({ train_indices });
	torch::Tensor X_test = X.index({ test_indices });
	torch::Tensor y_test = y.index({ test_indices });


	std::cout << "Train set: X_train shape = " << X_train.sizes() << ", y_train shape = " << y_train.sizes() << std::endl;
	std::cout << "Test set: X_test shape = " << X_test.sizes() << ", y_test shape = " << y_test.sizes() << std::endl;

	return std::tuple(X_train, y_train, X_test, y_test);

}


// Training function !!!

std::vector<float> TrainingFunction(PimaDiabetics& model, torch::Tensor X_train, torch::Tensor y_train, torch::nn::BCELoss loss_fn, torch::optim::Adam& optimizer, int batch_size) {


	int epochs = 100;
	batch_size = batch_size;
	std::vector <float> track_loss;
	model.train();

	for (int epoch = 0; epoch <= epochs; epoch++) {
		float train_loss = 0;
		for (int i = 0; i <= X_train.size(0); i += batch_size) {

			torch::Tensor X = X_train.slice(0, i, i + batch_size);
			torch::Tensor y = y_train.slice(0, i, i + batch_size);
			optimizer.zero_grad();
			torch::Tensor output = model.forward(X);
			torch::Tensor loss = loss_fn(output, y);
			loss.backward();
			optimizer.step();
			train_loss += loss.item<float>();
		}

		train_loss /= X_train.size(0) / batch_size;
		track_loss.push_back(train_loss);
		std::cout << "epochs :" << epoch << "\tloss :" << train_loss << std::endl;
	}
	return track_loss;
}

// Testing function !!!

void TestingFunction(PimaDiabetics& model, torch::Tensor X_test, torch::Tensor y_test, torch::nn::BCELoss loss_fn) {

	model.eval();
	torch::NoGradGuard InferenceMode();
	float test_loss = 0;
	torch::Tensor pred = model.forward(X_test);
	auto loss = loss_fn(pred, y_test);
	test_loss = loss.item<float>();
	auto correct = (pred.round() == y_test).sum();
	auto accuracy = correct / X_test.size(0);
	std::cout << "Loss : " << test_loss << "\tAccuracy :" << accuracy.item() << std::endl;

}


// Adding gaussian noise function !!!

std::pair<torch::Tensor, torch::Tensor> AddNoise(torch::Tensor X_train, torch::Tensor y_train) {
	torch::Tensor data = X_train.slice(1, 1, 8).clone();
	float std = 0.2;
	float mean = 0.1;
	torch::Tensor noise = torch::randn_like(data) * std + mean;
	torch::Tensor noisyXData = data + noise;
	noisyXData = torch::concat({ X_train.slice(1, 0, 1).clone(), noisyXData}, 1);
	noisyXData = torch::concat({ noisyXData, X_train.clone() }, 0);
	torch::Tensor ytrain = torch::concat({ y_train, y_train }, 0);


	return std::make_pair(noisyXData, ytrain);
}