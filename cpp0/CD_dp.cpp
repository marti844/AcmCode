#include<iostream>
#include<string.h>
using namespace std;
int n,m,a[25],dp[10001],vis[25][10001];

int main()
{
    while(cin>>n>>m)
    {
        memset(a,0,sizeof(a));
        memset(dp,0,sizeof(dp));
        memset(vis,0,sizeof(vis));
        for(register int i=1;i<=m;i++)
        {
            cin>>a[i];
        }
        for(int i=1;i<=m;i++)
        {
            for(int j=n;j>=a[i];j--)
            {
                if(dp[j]<dp[j-a[i]]+a[i])
                {
                    dp[j]=dp[j-a[i]]+a[i];
                    vis[i][j]=1;
                }
            }
        }
        for(int i=1,j=n;i<=m;i++)
        {
            if(vis[i][j]==1)
            {
                cout<<a[i]<<" ";
                j-=a[i];
            }
            
        }
        cout<<"sum:"<<dp[n]<<endl;
    }
}