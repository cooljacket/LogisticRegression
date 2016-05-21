#ifndef __LOGISTIC_REGRESSION_H__
#define __LOGISTIC_REGRESSION_H__

#include <vector>
#include <fstream>
#include <cmath>
using namespace std;

typedef vector<vector<int> > DATA;

class LogisticRegression
{
	const static double ERROR = 1e-10, INF = 1e15;
	vector<double> theta;
	DATA train_x, test_x;
	vector<double> train_y, h_prec;
	double alpha, costBound;
	int ColNum, Max_Iteration;

public:
	LogisticRegression(const char* fileName, double alpha, int MAX_Iter=1, double costBound=0.01);
	void readTrainData(fstream& in);
	double evaluate(const vector<int>& row);
	void predict();
	void gradientDescent(const DATA& X);
	double cost();
	double trainHelper(const DATA& X, const vector<double>& Y);
	double BatchTrain();
	double sigmoid(double x);
	bool cmpFloat(double x, double y);
};

#endif