#pragma once

#include "Task_2.h"

// Class definition !!!!

class PimaDiabetics : public torch::nn::Module {
public:
	PimaDiabetics(int input_size, int h1, int h2, int output_size) :
		input(input_size, h1),
		hidden(h1, h2),
		output(h2, output_size) {

		register_module("input", input);
		register_module("hidden", hidden);
		register_module("output", output);

	}
	torch::Tensor forward(torch::Tensor x) {

		x = torch::relu(input(x));
		x = torch::relu(hidden(x));
		x = torch::sigmoid(output(x));

		return x;
	}

	torch::nn::Linear input{ nullptr };
	torch::nn::Linear hidden{ nullptr };
	torch::nn::Linear output{ nullptr };
};