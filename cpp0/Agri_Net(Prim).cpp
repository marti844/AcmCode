#include<iostream>
#include<string.h>
#include<algorithm>
#include<math.h>

using namespace std;
const int MAXN=1010;
const int INF=0x3f3f3f3f;

// struct Edge{
//     int nxt,val,to;
// }edge[MAXN];
// int head[MAXN],cnt;
int n;
int mp[110][110];

// void init(){
//     memset(head,-1,sizeof(head));
// }

// void add(int x,int y,int z){
//     edge[++cnt].val=z;
//     edge[cnt].to=y;
//     edge[cnt].nxt=head[x];
//     head[x]=cnt;
// }

int dist[MAXN];
bool st[MAXN];
int Prim(){
    memset(dist,0x3f3f3f3f,sizeof(dist));
    int res=0;
    for(int i=0;i<n;i++){
        int t=-1;
        for(int j=1;j<=n;j++){
            if(!st[j]&&(t==-1||dist[t]>dist[j]))t=j;
            st[t]=true;
            if(i&&dist[t]==INF)return INF;
            if(i)res+=dist[t];
        }
        for(int j=1;j<=n;j++)dist[j]=min(dist[j],mp[t][j]);
    }
    return res;
}

int main()
{
    while(cin>>n){
        for(int i=1;i<=n;i++){
            for(int j=1;j<=n;j++){
                cin>>mp[i][j];
            }
        }
        int ans=Prim();
        cout<<ans<<endl;
    }
    return 0;
}

