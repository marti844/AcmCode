#include<iostream>
#include<string.h>
using namespace std;

int n,m,spa[3405],val[3405],dp[12881];

int main()
{
    memset(spa,0,sizeof(spa));
    memset(val,0,sizeof(val));
    memset(dp,0,sizeof(dp));
    cin>>n>>m;
    for(int i=1;i<=n;i++)
    {
        cin>>spa[i]>>val[i];
    }
    for(int i=n;i>=1;i--)
    {
        for(int j=m;j>=spa[i];j--)
        {
            if(j>=spa[i])
            {
                dp[j]=max(dp[j],dp[j-spa[i]]+val[i]);
            }
        }
    }
    
    cout<<dp[m]<<endl;
    return 0;
}
