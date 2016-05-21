out : main.o
	g++ -o out main.o

main.o : main.cpp LogisticRegression.o
LogisticRegression.o: LogisticRegression.h

clean:
	rm *.o

# g++ main.cpp LogisticRegression.cpp -o out
