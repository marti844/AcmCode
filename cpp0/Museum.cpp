#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
using namespace std;
typedef long long ll;
const int N = 900000+10;
int pre[N],vis[N],a[N],flag[N];
int main()
{
	int n,i,j,Max;
	ll t,ans,sum;
	sum=0;
	scanf("%d",&n);
		Max=0;
		for(i=1;i<=n;i++) {
			scanf("%d",&a[i]);
			Max=max(Max,a[i]);
			vis[a[i]]++;
			sum+=a[i];
		}
		sort(a+1,a+1+n);
		for(i=1;i<=N;i++) {
			pre[i]=pre[i-1];
			pre[i]+=vis[i];
		}
		ans=0;
		if(a[1]==1) ans=sum;
		else
		for(i=1;i<=n;i++) {
			if(flag[a[i]]) continue;
			flag[a[i]]=1;
			t=0;
			for(j=a[i]+a[i];j<=Max;j=j+a[i]) {
				flag[j]=1;
				t+=(ll)(pre[j-1]-pre[j-a[i]-1])*(j-a[i]);
			}
			t+=(ll)(pre[j-1]-pre[j-a[i]-1])*(j-a[i]);
			ans=max(ans,t);
		}
		printf("%I64d\n",ans);
 
	return 0;
}