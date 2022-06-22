#include<stdio.h>
#include<cstring>
#include<map>
#include<iostream>
using namespace std;
const int MAXN=1e6+10;
const int P=223;

typedef unsigned long long ull;

ull pre[MAXN],p[MAXN];
ull get(int l,int r)
{
    return pre[r]-pre[l-1]*p[r-l+1];
}

int main()
{
    int a[]={0,1,2,3};
    char c;
    cin>>c;
    cout<<a[c-97];
    return 0;
}