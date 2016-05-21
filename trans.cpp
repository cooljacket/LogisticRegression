#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

int main() {
	fstream in("data/train_70.txt", ios::in);
	fstream out("data/train.txt", ios::out);

	int category, index, value, cnt = 0;
	char ch;
	string line;
	while (in >> category) {
		out << category;
		getline(in, line);
		stringstream ss(line);
		while (ss >> index >> ch >> value)
			out << ' ' << index;
		out << endl;
		++cnt;
	}

	in.close();
	out.close();

	cout << "Total " << cnt << " samples" << endl;

	return 0;
}