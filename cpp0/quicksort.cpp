#include<iostream>
#include<cmath>
#include<string.h>
#include<algorithm>
#include<vector>
#include<functional>
#include<queue>

using namespace std;
const int MAXN=200010;
const int INF=0x3f3f3f3f;
typedef long long ll;
typedef pair<int,int> pii;

struct Node{
	int lc,rc;
	int minval,maxval;
}tree[MAXN*4];

void pushUp(int k){

	tree[k].minval=min(tree[2*k].minval, tree[2*k+1].minval);
	tree[k].maxval=max(tree[2*k].maxval, tree[2*k+1].maxval);
	
}
void build_tree(int k,int l,int r){
	tree[k].lc=l;
	tree[k].rc=r;
	if(l==r){
		return;
	}
}

int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);
	
	return 0;
}







