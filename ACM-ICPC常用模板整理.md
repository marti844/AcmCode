# ACM-ICPC常用模板整理

<p align="right"> by 石珂安 </p>

## 读入输出

```c++
inline int read(){
    int x=0,f=1;char ch=getchar();
    while(!isdigit(ch)){
        if(ch=='-'){
            f=-1;
            ch=getchar();
        }
    }
    while(isdigit(ch)){
        x=x*10+ch-48;
        ch=getchar();
    }
    return x*f;
}

inline void print(int x)
{
	if(x==0)return;
	print(x/10);
	putchar(x%10+'0');
}
```

## 排序

### 桶排序

```c++
int* sort_array(int *arr, int n) {
    int i;
	int maxValue = arr[0];
	for (i = 1; i < n; i++) 
		if (arr[i] > maxValue)  // 输入数据的最大值
			maxValue = arr[i]; 
	
	// 设置10个桶，依次0，1，，，9
	const int bucketCnt = 10;
	vector<int> buckets[bucketCnt];
	// 桶的大小bucketSize根据数组最大值确定：比如最大值99， 桶大小10
	// 最大值999，桶大小100
	// 根据最高位数字映射到相应的桶，映射函数为 arr[i]/bucketSize
	int bucketSize = 1;
	while (maxValue) {		//求最大尺寸 
		maxValue /= 10;
		bucketSize *= 10;
	}
	bucketSize /= 10;		//桶的个数 
	// 入桶
	for (int i=0; i<n; i++) {
		int idx = arr[i]/bucketSize;			//放入对应的桶 
		buckets[idx].push_back(arr[i]);
		// 对该桶使用插入排序(因为数据过少，插入排序即可)，维持该桶的有序性
		for (int j=int(buckets[idx].size())-1; j>0; j--) {
			if (buckets[idx][j]<buckets[idx][j-1]) {
				swap(buckets[idx][j], buckets[idx][j-1]);
			}
		}
	}
	// 顺序访问桶，得到有序数组
	for (int i=0, k=0; i<bucketCnt; i++) {
		for (int j=0; j<int(buckets[i].size()); j++) {
			arr[k++] = buckets[i][j];
		}
	}
	return arr;
}
```

## 数据结构

### 树状数组

```c++
// C++ Version
int t1[MAXN], t2[MAXN], n;

inline int lowbit(int x) { return x & (-x); }

void add(int k, int v) {
  int v1 = k * v;
  while (k <= n) {
    t1[k] += v, t2[k] += v1;
    k += lowbit(k);
  }
}

int getsum(int *t, int k) {
  int ret = 0;
  while (k) {
    ret += t[k];
    k -= lowbit(k);
  }
  return ret;
}

void add1(int l, int r, int v) {
  add(l, v), add(r + 1, -v);  // 将区间加差分为两个前缀加
}

long long getsum1(int l, int r) {
  return (r + 1ll) * getsum(t1, r) - 1ll * l * getsum(t1, l - 1) -
         (getsum(t2, r) - getsum(t2, l - 1));
}
```

### 树链剖分+线段树维护

```c++
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N = 100050;

int a[N];						   //线段树的基础数组
int n, m;						   // n为节点数，m为操作数
int op;							   //操作
int fa[N], dep[N], siz[N], son[N]; // fa记录父节点，dep记录深度，siz记录子树节点数，son记录重儿子
int head[N];
int tim, dfn[N], top[N], v[N]; // tim为时间戳，dfn为节点在a数组的下标，top记录重链其实轻儿子的下标，v记录权值
int cnt;
int mod; //取模
int r;	 //以哪个节点为根节点

struct edge
{
	int t, nxt;
} e[N];

void init()
{
	memset(head, -1, sizeof head);
	cnt = 0;
	tim = 0;
}

void add(int u, int v)
{
	e[++cnt].t = v;
	e[cnt].nxt = head[u];
	head[u] = cnt;
}
struct Node
{
	int l, r;
	int sum1;
	int sum2;
	int t1, t2;
	int maxn;
	int minn;

} tr[N << 2];

void pushup(Node &u, Node &l, Node &r)
{
	u.sum1 = l.sum1 + r.sum1;
	u.sum2 = l.sum2 + r.sum2;
	u.maxn = max(l.maxn, r.maxn);
	u.minn = min(l.minn, r.minn);
}
void pushup(int u)
{ // 由子节点的信息，来计算父节点的信息
	pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}

void eval(Node &t, int add, int mul)
{
	t.sum2 = t.sum2 * mul * mul;
	t.sum2 += 2 * add * t.sum1 * mul + add * add * (t.r - t.l + 1);
	t.sum1 = mul * t.sum1 + add * (t.r - t.l + 1);
	t.t2 *= mul;
	t.t1 = t.t1 * mul + add;
}

//懒标记
void pushdown(int u)
{
	eval(tr[u << 1], tr[u].t1, tr[u].t2);
	eval(tr[u << 1 | 1], tr[u].t1, tr[u].t2);
	tr[u].t1 = 0, tr[u].t2 = 1;
}

// 节点tr[u]存储区间[l, r]的信息
void build(int u, int l, int r)
{
	if (l == r)
	{
		tr[u] = {l, r, a[l], a[l] * a[l], 0, 1, a[l], a[l]};
	} //储存信息
	else
	{
		tr[u] = {l, r, 0, 0, 0, 1};
		int mid = l + r >> 1;
		build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
		pushup(u);
	}
}

int querysum1(int u, int l, int r) //求和
{
	auto &t = tr[u];
	if (t.l >= l && t.r <= r)
		return t.sum1 % mod;
	else
	{
		pushdown(u);
		int mid = t.l + t.r >> 1;
		int res = 0;
		if (l <= mid)
			res = querysum1(u << 1, l, r) % mod;
		if (r > mid)
			res += querysum1(u << 1 | 1, l, r) % mod;
		pushup(u);
		return res % mod;
	}
}

int querysum2(int u, int l, int r) //求平方和
{
	auto &t = tr[u];
	if (t.l >= l && t.r <= r)
		return t.sum2 % mod;
	else
	{
		pushdown(u);
		int mid = t.l + t.r >> 1;
		int res = 0;
		if (l <= mid)
			res = querysum2(u << 1, l, r) % mod;
		if (r > mid)
			res += querysum2(u << 1 | 1, l, r) % mod;
		pushup(u);
		return res % mod;
	}
}
int querymax(int u, int l, int r)
{

	if (tr[u].l >= l && tr[u].r <= r)
		return tr[u].maxn; // 树中节点，已经被完全包含在[l, r]中了

	int mid = tr[u].l + tr[u].r >> 1;
	int v = -0x3f3f3f3f;
	if (l <= mid)
		v = querymax(u << 1, l, r);
	if (r > mid)
		v = max(v, querymax(u << 1 | 1, l, r)); // 右区间从mid+1开始

	return v;
}

int querymin(int u, int l, int r)
{

	if (tr[u].l >= l && tr[u].r <= r)
		return tr[u].minn; // 树中节点，已经被完全包含在[l, r]中了

	int mid = tr[u].l + tr[u].r >> 1;
	int v = 0x3f3f3f3f;
	if (l <= mid)
		v = querymin(u << 1, l, r);
	if (r > mid)
		v = min(v, querymin(u << 1 | 1, l, r)); // 右区间从mid+1开始

	return v;
}

void modify1(int u, int l, int r, int d) //每个元素乘上一个数
{
	if (tr[u].l >= l && tr[u].r <= r)
		eval(tr[u], 0, d);
	else
	{ // 一定要分裂
		pushdown(u);
		int mid = tr[u].l + tr[u].r >> 1;
		if (l <= mid)
			modify1(u << 1, l, r, d);
		if (r > mid)
			modify1(u << 1 | 1, l, r, d);
		pushup(u);
	}
}

void modify2(int u, int l, int r, int d) //每个元素加上一个数
{
	if (tr[u].l >= l && tr[u].r <= r)
		eval(tr[u], d, 1);
	else
	{ // 一定要分裂
		pushdown(u);
		int mid = tr[u].l + tr[u].r >> 1;
		if (l <= mid)
			modify2(u << 1, l, r, d);
		if (r > mid)
			modify2(u << 1 | 1, l, r, d);
		pushup(u);
	}
}

void dfs1(int u, int f)
{
	fa[u] = f;
	dep[u] = dep[f] + 1;
	siz[u] = 1;
	int maxsize = -1;
	for (int i = head[u]; ~i; i = e[i].nxt)
	{
		int v = e[i].t;
		if (v == f)
			continue;
		dfs1(v, u);
		siz[u] += siz[v];
		if (siz[v] > maxsize)
		{
			maxsize = siz[v];
			son[u] = v;
		}
	}
}

void dfs2(int u, int t)
{
	dfn[u] = ++tim;
	top[u] = t;
	a[tim] = v[u];
	if (!son[u])
		return;
	dfs2(son[u], t);
	for (int i = head[u]; ~i; i = e[i].nxt)
	{
		int v = e[i].t;
		if (v == fa[u] || v == son[u])
			continue;
		dfs2(v, v);
	}
}

void mson(int x, int z) //把以x为根的子树所有值加z
{
	modify2(1, dfn[x], dfn[x] + siz[x] - 1, z);
}

int qson(int x) //查询以x为根节点的子树的权值和
{
	return querysum1(1, dfn[x], dfn[x] + siz[x] - 1);
}

void mchain(int x, int y, int z) // x到y的最短路径上的所有节点权值加z
{
	z %= mod;
	while (top[x] != top[y])
	{
		if (dep[top[x]] < dep[top[y]])
			swap(x, y);
		modify2(1, dfn[top[x]], dfn[x], z);
		x = fa[top[x]];
	}
	if (dep[x] > dep[y])
		swap(x, y);
	modify2(1, dfn[x], dfn[y], z);
}

int qchain(int x, int y) // x到y最短路径的所有节点权值和
{
	int ret = 0;
	while (top[x] != top[y])
	{
		if (dep[top[x]] < dep[top[y]])
			swap(x, y);
		ret += querysum1(1, dfn[top[x]], dfn[x]);
		x = fa[top[x]];
	}
	if (dep[x] > dep[y])
		swap(x, y);
	ret += querysum1(1, dfn[x], dfn[y]);
	return ret % mod;
}

signed main()
{
	scanf("%lld%lld%lld%lld", &n, &m, &r, &mod);
	for (int i = 1; i <= n; i++)
		scanf("%lld", &v[i]);
	int x, y, z;
	init();
	for (int i = 1; i <= n - 1; i++)
	{
		scanf("%lld%lld", &x, &y);
		add(x, y);
		add(y, x);
	}
	dfs1(r, r);
	dfs2(r, r);
	build(1, 1, n);
	for (int i = 1; i <= m; i++)
	{
		int op;
		cin >> op;
		if (op == 1)
		{
			cin >> x >> y >> z;
			mchain(x, y, z);
		}
		else if (op == 2)
		{
			cin >> x >> y;
			cout << qchain(x, y) << endl;
		}
		else if (op == 3)
		{
			cin >> x >> z;
			mson(x, z);
		}
		else
		{
			cin >> x;
			cout << qson(x) << endl;
		}
	}
	// for(int i=1;i<=n;i++) cout<<a[i]<<endl;
	system("pause");
	return 0;
}
```

### 序列分块

### 莫队

### 倍增思想（LCA、ST表）

```c++
// ST表
const int maxn = 100000;
int ST[maxn][22];
int n;
void build()
{
    for (int j = 1; j <= 21;j++)
    {
        for (int i = 1; i + (1 << j) - 1 <= n;i++)
            ST[i][j] = max(ST[i][j - 1], ST[i + (1 << (j - 1))][j - 1]);
    }
}
int query(int l,int r)
{
    int s = (int)log2(r - l + 1);
    return max(ST[l][s], ST[r - (1 << l) + 1][s]);
}
int main()
{
    cin >> n;
    for (int i = 1; i <= n;i++)
        cin >> ST[0][i];
}
```

### 舞蹈链

### 主席树

### 红黑树

## 图论

### 存图

```c++
struct Edge{
  int to,val,nxt;
}edge[maxn];
int head[maxn],cnt;

inline void init(){
  memset(head,-1,sizeof head);
  cnt=0;
}

inline void addedge(int s,int t,int w){
  edge[++cnt].to=t;
  edge[cnt].val=w;
  edge[cnt].nxt=head[s];
  head[s]=cnt;
}
```

### 最短路算法

#### Floyed

#### Dijkstra

```c++
void dijkstra(int n)
{
  	typedef pair<int, int> pii;
    priority_queue<pii,vector<pii>,greater<pii> >q;
    memset(dis,0x3f,sizeof dis);
    q.push({0,1});
    dis[1]=0;
    while(!q.empty())
    {
        pii temp = q.top();//记录堆顶，即堆内最小的边并将其弹出 
        q.pop();
        int u = temp.second;//点 
        if(vis[u]) continue;//如果被访问过，就跳过 
        vis[u]=true;//标记 
        for(int i = head[u];i!=-1;i=edge[i].next)//搜索堆顶的所有连边 
        {
            int v = edge[i].v;
            if(dis[v]>dis[u]+edge[i].w)//松弛操作 
            {
                dis[v]=dis[u]+edge[i].w;
                q.push({dis[v],v});//把新遍历的点加到堆中 
            }
        }
    }
    //for(int i=1;i<=n;i++) cout<<dis[i]<<endl;
}
```

#### SPFA

```c++
void spfa()
{
    for(int i=1;i<=n;i++)
    dist[i]=INT_MAX;//题目要求初始化为2^31-1即int整型的最大范围

    dist[s]=0;
    queue<int>q;
    q.push(s);
    st[s]=true;

    int t;
    while (!q.empty())
    {
        t=q.front();
        q.pop();
        st[t]=false;

        for(int i=h[t];i!=-1;i=ne[i])
        {
            int j=e[i];
            if(dist[j]>dist[t]+w[i])
            {
                dist[j]=dist[t]+w[i];
                if(!st[j])
                {
                    q.push(j);
                    st[j]=true;
                }
            }
        }
    }    
}
```

### 最小生成树

#### Prim

```c++
int prim(int n)//n为顶点个数
{
		typedef pair<int, int> pii;
		priority_queue<pii, vector<pii>, greater<pii> > q;
		int dis[10010],vis[10010],ans=0;
		memset(dis,0x3f,sizeof dis);
		memset(vis,0,sizeof vis);
		dis[1]=0;
		q.push(make_pair(0,1));
		while(!q.empty()){
				int f=q.top().first;
				int s=q.top().second;
				q.pop();
				if(vis[s])continue;
				sum++;ans+=f;vis[s]=1;
				for(int i=head[s];~i;i=edge[i].nxt){
						if(edge[i].val<dis[edge[i].to]&&vis[edge[i].to]){
								dis[edge[i].to]=edge[i].val;
								q.push(make_pair(edge[i].val,edge[i].to));
						}
				}
    }
    return ans;
}
```

#### Kruskal

```c++
int Kruskal()
{
    int sum=0;
    sort(edge+1,edge+r+1,cmp);
    for(int i=1;i<=n;i++)fa[i]=i;
    for(int i=1;i<=r;i++)
    {
        int f=edge[i].from;
        int t=edge[i].to;
        int va=edge[i].val;
        f=find(f);
        t=find(t);
        if(f!=t)
        {
            fa[f]=t;
            sum+=va;
        }
    }
    return sum;
}
```

### 网络流

#### Dinic

```c++
// 注意建图操作cnt从1开始！！！建图的同时建边权为零的反图
int dep[maxn],no[maxn];// 弧优化

inline bool bfs(){
	for(int i=1;i<=n;++i)dep[i]=INF;
	dep[s]=0;
	queue<int> q;
	q.push(s);
	no[s]=head[s];
	while(!q.empty()){
		int now=q.front();
		q.pop();
		for(int i=head[now];~i;i=edge[i].nxt){
			int to=edge[i].to;
			if(edge[i].val>0&&dep[to]==INF){
				q.push(to);
				no[to]=head[to];
				dep[to]=dep[now]+1;
				if(to==t)return true;
			}
		}
	}
	return false;
}

inline int dinic(int x,ll flow){
	if(x==t)return flow;
	ll tmp,sum=0;
	for(int i=no[x];(~i)&&flow;i=edge[i].nxt){
		no[x]=i;
		int to=edge[i].to;
		if(edge[i].val>0&&(dep[to]==dep[x]+1)){
			tmp=dinic(to,min(flow,edge[i].val));
			if(tmp==0)dep[to]=INF; //剪枝
			flow-=tmp;
			sum+=tmp;
			edge[i].val-=tmp;
			edge[i^1].val+=tmp;
		}
	}
	// if(sum==0)return dep[x]=0;
	return sum;
}
```

## 数论

### 快速幂

```c++
// C++ Version
long long ksm(long long a, long long b, int c) {
  long long res = 1;
  a%=c;
  while (b > 0) {
    if (b & 1) res = res * a % c;
    a = a * a % c;
    b >>= 1;
  }
  return res;
}
```

### gcd/lmp

```c++
int gcd(int a, int b){
   return a % b ? gcd(b, a % b) : b;
}
int lmp(int a,int b){
   return a*b/gcd(a,b);
}
```

### 欧拉筛

```c++
#define n 10000

bool vis[n];//标记
int prim[n];//储存素数
int num=0;//素数数量

void getprim()
{
    memset(vis,true,sizeof(vis));//初始化为全体素数
    vis[0]=vis[1]=false;//01不是素数
    for(int i=2;i<=n;i++){
        if(vis[i]) prim[++num]=i;
        for(int j=1;j<=num&&i*prim[j]<=n;j++){//合数在给定范围内
            vis[i*prim[j]]=false;
            if(i%prim[j]==0) break;
        }
    }
}

```

## 字符串

### 字符串哈希

```c++
const int M = 1e9 + 7;
const int B = 233;

typedef long long ll;

int get_hash(const string& s) {
  int res = 0;
  for (int i = 0; i < s.size(); ++i) {
    res = (ll)(res * B + s[i]) % M;
  }
  return res;
}

bool cmp(const string& s, const string& t) {
  return get_hash(s) == get_hash(t);
}
```

### KMP

```c++
const int MAXN=1000100;
    char s[MAXN],t[MAXN];//s是主串；t是模式串
    int m,n;//m为模式串长度；n为主串长度
    int nxt[MAXN];

    void kmp_pre()
    {
        int i,j;
        j=nxt[0]=-1;
        i=0;
        while(i<m)
        {
            if(j==-1||t[i]==t[j])nxt[++i]=++j;
            else j=nxt[j];
        }
    }
    //出现次数kmp
    int kmp_count()
    {
        int i=0,j=0;
        int ans=0;
        if(m==1&&n==1)
        {
            if(t[0]==s[0]) return 1;
            else return 0;
        }
        kmp_pre();
        for(i=0;i<n;i++)
        {
            while(j>0&&s[i]!=t[j])j=nxt[j];
            if(s[i]==t[j])j++;
            if(j==m)
            {
                ans++;
                j=nxt[j];
            } 
        }
        return ans;
    }
    //返回位置kmp
    int KMP(){
 
    int i=0;
    int j=0;
 
    while(i<n&&j<m){
        if(j==-1||t[i]==p[j]){
            i++;
            j++;
            if(j==m&&i!=n){//当模式串到达结尾时，回到指定位置
                j=nxt[j];
            }
        }
        else{
           j=nxt[j];
        }
    }
 
    return j;//返回前缀的位置
}
```

### Manacher

```c++
vector<int> d1(n), d2(n);
for (int i = 0; i < n; i++) {
  d1[i] = 1;
  while (0 <= i - d1[i] && i + d1[i] < n && s[i - d1[i]] == s[i + d1[i]]) {
    d1[i]++;
  }

  d2[i] = 0;
  while (0 <= i - d2[i] - 1 && i + d2[i] < n &&
         s[i - d2[i] - 1] == s[i + d2[i]]) {
    d2[i]++;
  }
}
```

### AC自动机



### Tire树

```c++
// C++ Version
struct trie {
  int nex[100000][26], cnt;
  bool exist[100000];  // 该结点结尾的字符串是否存在

  void insert(char *s, int l) {  // 插入字符串
    int p = 0;
    for (int i = 0; i < l; i++) {
      int c = s[i] - 'a';
      if (!nex[p][c]) nex[p][c] = ++cnt;  // 如果没有，就添加结点
      p = nex[p][c];
    }
    exist[p] = 1;
  }

  bool find(char *s, int l) {  // 查找字符串
    int p = 0;
    for (int i = 0; i < l; i++) {
      int c = s[i] - 'a';
      if (!nex[p][c]) return 0;
      p = nex[p][c];
    }
    return exist[p];
  }
};
```

### 后缀自动机



