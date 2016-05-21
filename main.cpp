#include <stdio.h>
#include <ctime>
#include "LogisticRegression.h"

int main() {
	clock_t start = clock();
	LogisticRegression lg("data/train.txt", 0.125, 50);
	lg.BatchTrain();
	printf("Totol cost time %lfs\n", (clock()-start)*1.0/CLOCKS_PER_SEC);
	
	// vector<double> result(lg.Test());
	// do something more about the predict result...

	return 0;
}
