#include<iostream>
#include<algorithm>
#include<cmath>
#include<string.h>
#include<queue>
#include<stack>
#include<map>
#include<functional>
#include<vector>

using namespace std;
typedef pair<int,int> pii;
typedef unsigned long long ull;
typedef long long ll;
const int maxn = 1e5+10;
const int INF = 0x3f3f3f3f;

int arr[maxn],dp1[maxn],dp2[maxn];

inline int read()
{
	int x=0,y=1;char c=getchar();
	while (c<'0'||c>'9') {if (c=='-') y=-1;c=getchar();}
	while (c>='0'&&c<='9') x=x*10+c-'0',c=getchar();
	return x*y;
}

int main()
{
	ios::sync_with_stdio(0);cout.tie(0);
	int len = 0;
	while(cin>>arr[len]){
		len++;
	}
//	len++;
	memset(dp1,0x1f,sizeof dp1);
	int maxx = dp1[0];
	for(int i = 0;i < len;++i){
		*lower_bound(dp1,dp1+len,arr[i]) = arr[i];
	}
	int num = 0;
	while(dp1[num]!=maxx)++num;
	for(int i = 0;i < (len>>1);++i){
		swap(arr[i], arr[len-i-1]);
	}
	memset(dp2,0x1f,sizeof dp2);
	int mx = dp2[0];
	for(int i = 0;i < len;++i){
		*upper_bound(dp2,dp2+len,arr[i]) = arr[i];
	}
	int ans = 0;
	while(dp2[ans]!=mx)++ans;
	cout<<ans<<"\n"<<num;
	return 0;
	
}







