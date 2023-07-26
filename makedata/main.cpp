#include"makedata.h"
// #define int long long
#define pii pair<int, int>
using namespace std;
const int N = 1e6 + 10;
int n, m, k = 1;
vector<int> G[N];
int deg[N];
queue<int> q;
map<pii, int> mp;
vector<int> ans;

inline void init() {
    k = 1;
    for(int i = 1; i <= N; ++i) {
        G[i].clear();
    }
    memset(deg, 0, sizeof deg);
    while(!q.empty()) q.pop();
    mp.clear();
    ans.clear();
}

inline void topsort() {
    while(!q.empty()) {
        int u = q.front();
        ans.push_back(u);
        q.pop();
        for(auto v : G[u]) {
            deg[v] --;
            if(deg[v] == 0) q.push(v);
        }
    }
    if(ans.size() < n) {
        k = 2;
    }
}

inline void solve() {

    for(int i = 1; i <= n; ++i) {
        if(deg[i] == 0) {
            q.push(i);
        }
    }
    topsort();

    outfile << k << endl;
    
    if(k == 1){
        for(int i = 0; i < n; ++i) {
            outfile << ans[i] << " ";
        }
    }
    if(k == 2) {
        for(int i = 1; i <= n; ++i) {
            outfile << i << " ";
        }
        outfile << endl;
        for(int i = n; i >= 1; --i) {
            outfile << i << " ";
        }
    }  
    return;
}

void make(int tp){

    times = 1;
    // 生成的数据范围
    int T = 1000;
    infile << T << endl;
    while(T -- ){
        init();

        m = num(1, 20, 0); n = num(1, 20, 0);
        infile << n << " " << m << endl;
        
        for(int i = 1; i <= m; ++i) {
            int s = num(1, 20, 0), t = num(1, 20, s);
            infile << s << " " << t << endl;
            G[s].push_back(t);
            deg[t] ++;
        }
            
        //以上按题意读入 
        
        /*
        暴力求解的代码
        */     
        // ios::sync_with_stdio(0);
        // cin.tie(0), cout.tie(0);

        // int T;
        // T = 1000;
        // infile << T << endl;
        // while(T --) {
        //     init();
        //     solve();
        //     outfile << endl;

        // }
        
        solve();
        outfile << endl;
    }
    return ;
}
