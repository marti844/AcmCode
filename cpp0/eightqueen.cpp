#include<iostream>
#include<vector>
#include<string.h>
#include<cmath>
using namespace std;
typedef long long ll;
ll a[100]={0},b[100]={0},c[100]={0},flag[100]={0};//用一维数组表示行列、斜方向上有无棋子
ll n;
vector<int> v;

void sum()
{
	int num=0;
	for(int i=8;i>0;i--)
	{
		num+=flag[i]*pow(10,(8-i));
	}
	v.push_back(num);
}

void queen(int j)
{
	
	for(int i=1;i<=8;i++)
	{
		if(!a[i]&&!(b[i+j])&&!(c[i-j+7]))//判断能否放置
		{
			a[i]=1;
			b[i+j]=1;
			c[i-j+7]=1;
			flag[j]=i;
			if(j==8)sum();
			else queen(j+1);
			a[i]=0;//回溯
			b[i+j]=0;
			c[i-j+7]=0;
		}
	}
	return;
}

int main()
{
	int n,x;
	cin>>n;
	queen(1);
	for(int i=0;i<n;i++)
	{
		cin>>x;
		cout<<v[x-1]<<endl;
	}
	return 0;
}