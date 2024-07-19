
#pragma once
#include "Task_2.h"

std::vector<std::vector<float>> readCSV(const std::string& filename);
std::tuple <torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TrainTestSplit(torch::Tensor X, torch::Tensor y);
std::vector<float> TrainingFunction(PimaDiabetics& model, torch::Tensor X_train, torch::Tensor y_train, torch::nn::BCELoss loss_fn, torch::optim::Adam& optimizer, int batch_size);
void TestingFunction(PimaDiabetics& model, torch::Tensor X_test, torch::Tensor y_test, torch::nn::BCELoss loss_fn);
std::pair<torch::Tensor, torch::Tensor> AddNoise(torch::Tensor X_train, torch::Tensor y_train);

int main()
{
	
	// Reading the csv dataset file
	std::string filename = "file_path";
	std::vector<std::vector<float>> dataset = readCSV(filename);

	int n = dataset.size();
	int m = dataset[0].size();
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto tensor = torch::zeros({ n,m }, options);
	for (int i = 0; i < n; i++)
		tensor.slice(0, i, i + 1) = torch::from_blob(dataset[i].data(), { m }, options);



	//std::cout << "Tensor:\n" << tensor << std::endl;

	std::cout << "Type of vec: " << typeid(dataset).name() << std::endl;
	std::cout << "Type of tensor: " << typeid(tensor).name() << std::endl;
	//check_1();

	// Slice the dependant and independant feautres
	torch::Tensor X = tensor.slice(1, 0, 8);
	torch::Tensor y = tensor.slice(1, 8);
	std::cout << std::endl;
	//std::cout << "Input features :\n" << X << std::endl;
	//std::cout << "Output featiures :\n" <<y << std::endl;

	// Train test split
	auto [X_train, y_train, X_test, y_test] = TrainTestSplit(X, y);


	torch::manual_seed(42);
	// 42 0002 // 42 0.001 // 
	// Create model instance
	PimaDiabetics model(8, 12, 10, 1);
	std::cout << model << std::endl;

	// Setting loss function and oprtimizer
	auto loss_fn = torch::nn::BCELoss();
	//auto optimizer1 = torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(0.01));
	torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
	int batch_size = 10;
	std::vector <float> loss;
	// Train the model
	loss = TrainingFunction(model, X_train, y_train, loss_fn, optimizer, batch_size);

	//std::cout << loss << std::endl;

	// Testing the model
	TestingFunction(model, X_test, y_test, loss_fn);
	
	// Adding gaussian noise

	auto[Xtrain, ytrain] = AddNoise(X_train, y_train);
	std::cout << "shape of x noise data :" << Xtrain.sizes() << std::endl;
	std::cout << "shape of y noise data :" << ytrain.sizes() << std::endl;

	torch::manual_seed(32);

	// Creating new model instance
	PimaDiabetics newModel(8, 12, 10, 1);
	auto criterion = torch::nn::BCELoss();
	torch::optim::Adam optimizer1(newModel.parameters(), torch::optim::AdamOptions(0.002));

	std::vector<float> noisyloss;

	//Train and test the new model
	noisyloss = TrainingFunction(newModel, Xtrain, ytrain, criterion, optimizer1, batch_size= 15);
	std::cout << std::endl;
	TestingFunction(newModel, X_test, y_test, criterion);

	std::cin.get();

	return 0;
}
