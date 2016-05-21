#include "LogisticRegression.h"
#include <string>
#include <sstream>
#include <stdio.h>
#include <ctime>
#include <algorithm>
using namespace std;


LogisticRegression::LogisticRegression(const char* fileName, double alpha, int MAX_Iter, double costBound)
: alpha(alpha), costBound(costBound), Max_Iteration(MAX_Iter)
{
	fstream data(fileName, ios::in);
	readTrainData(data);
	data.close();
	++ColNum;
	theta = vector<double>(ColNum);
	h_prec = vector<double>(train_x.size());
}


void LogisticRegression::readTrainData(fstream& in) {
	string line;
	int category, value;
	clock_t start = clock();

	getline(in, line);
	printf("Reading...\n");
	while (getline(in, line)) {
		if (line.empty())
			continue;
		stringstream ss(line);
		ss >> category;
		train_y.push_back(category);

		vector<int> row(1, 0);
		while (ss >> value) {
			row.push_back(value);
			ColNum = max(value, ColNum);
		}
		train_x.push_back(row);
		// if (train_x.size() >= 500000)
		// 	break;
	}
	printf("Read total %d samples, use time %lfs\n", int(train_x.size()), (clock()-start)*1.0/CLOCKS_PER_SEC);
}


// 计算一行的预测值h(x) = sum(theta[i] * x[i])
// 但是并没有存整一行x，而是存了row，即那些值为1的下标的数组
// 那么只需要把这些下标对应的theta加起来就是相同的结果了
double LogisticRegression::evaluate(const vector<int>& row) {
	double h = 0;
	for (int i = 0; i < row.size(); ++i) {
		h += theta[row[i]];	// theta[row[i]] * 1.0
	}

	return sigmoid(h);
}


// 为所有的sample预测值
void LogisticRegression::predict() {
	for (int i = 0; i < train_x.size(); ++i)
		h_prec[i] = evaluate(train_x[i]);
}


// 批量梯度下降，注意theta_delta的作用就好了
// 即theta_delta[X[i][j]] += (h_prec[i] - train_y[i]) * 1.0;
void LogisticRegression::gradientDescent(const DATA& X) {
	vector<double> theta_delta(ColNum, 0.0);
	for (int i = 0; i < X.size(); ++i) {
		for (int j = 0; j < X[i].size(); ++j) 
			theta_delta[X[i][j]] += h_prec[i] - train_y[i];
	}
	
	for (int i = 0; i < ColNum; ++i)
		theta[i] -= alpha * theta_delta[i] / X.size();
}


double LogisticRegression::cost() {
	double J = 0.0;
	for (int i = 0; i < train_x.size(); ++i) {
		// 因为log(0)是NAN，所以需要特殊判断一下
		if (cmpFloat(h_prec[i], 0))
			J += -train_y[i] * INF;
		else if (cmpFloat(h_prec[i], 1))
			J += -(1 - train_y[i]) * INF;
		else
			J += -train_y[i]*log(h_prec[i]) - (1-train_y[i])*log(1-h_prec[i]);
	}
	return J / train_x.size();
}


// 一次迭代里做的事
double LogisticRegression::trainHelper(const DATA& X, const vector<double>& Y) {
	predict();
	gradientDescent(X);
}


double LogisticRegression::BatchTrain() {
	int times = 0;

	while (times++ < Max_Iteration) {
		clock_t start = clock();
		trainHelper(train_x, train_y);
		double J = cost();
		printf("Training[%d], cost=%.6lf, use time %lfs\n", times, J, (clock()-start)*1.0/CLOCKS_PER_SEC);
		if (J < costBound)
			break;
	}

	predict();
	double J = cost();
	printf("Finally, cost=%.6lf\n", J);
	return J;
}


double LogisticRegression::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}


bool LogisticRegression::cmpFloat(double x, double y) {
	if (x >= y && x - y < ERROR)
		return true;
	if (y >= x && y - x < ERROR)
		return true;
	return false;
}