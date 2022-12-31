//makedata.h 2020.12.15
#include<bits/stdc++.h>
using namespace std;

ofstream infile,outfile;
int times=20;

void file(int tp){
	char inname[10],outname[11];
	infile.close();
	outfile.close();
	sprintf(inname,"data%02d.in",tp);
	sprintf(outname,"data%02d.out",tp);
	infile.open(inname);
	outfile.open(outname);
}

#define num(a,b) ((rand()*RAND_MAX+rand())%(a-b+1)+a)

void make(int tp);

int main(){
	srand(time(0));
	for(int tp=1;tp<=times;++tp){
	    file(tp);
	    make(tp);
	}
	return 0;
}
