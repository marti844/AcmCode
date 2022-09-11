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

### 选择排序

```c++
// C++ Version
void selection_sort(int* a, int n) {
  for (int i = 1; i < n; ++i) {
    int ith = i;
    for (int j = i + 1; j <= n; ++j) {
      if (a[j] < a[ith]) {
        ith = j;
      }
    }
    std::swap(a[i], a[ith]);
  }
}
```

### 插入排序

```c++
// C++ Version
void insertion_sort(int* a, int n) {
  // 对 a[1],a[2],...,a[n] 进行插入排序
  for (int i = 2; i <= n; ++i) {
    int key = a[i];
    int j = i - 1;
    while (j > 0 && a[j] > key) {
      a[j + 1] = a[j];
      --j;
    }
    a[j + 1] = key;
  }
}
```

### 希尔排序

```c++
// C++ Version
void shell_sort(int arr[], int len) {
    int gap, i, j;
    int temp;
    for (gap = len >> 1; gap > 0; gap >>= 1)
        for (i = gap; i < len; i++) {
            temp = arr[i];
            for (j = i - gap; j >= 0 && arr[j] > temp; j -= gap)
                arr[j + gap] = arr[j];
            arr[j + gap] = temp;
        }
}
```

### 堆排序

```c++
// C++ Version
void sift_down(int arr[], int start, int end) {
  // 计算父结点和子结点的下标
  int parent = start;
  int child = parent * 2 + 1;
  while (child <= end) {  // 子结点下标在范围内才做比较
    // 先比较两个子结点大小，选择最大的
    if (child + 1 <= end && arr[child] < arr[child + 1]) child++;
    // 如果父结点比子结点大，代表调整完毕，直接跳出函数
    if (arr[parent] >= arr[child])
      return;
    else {  // 否则交换父子内容，子结点再和孙结点比较
      swap(arr[parent], arr[child]);
      parent = child;
      child = parent * 2 + 1;
    }
  }
}

void heap_sort(int arr[], int len) {
  // 从最后一个节点的父节点开始 sift down 以完成堆化 (heapify)
  for (int i = (len - 1 - 1) / 2; i >= 0; i--) sift_down(arr, i, len - 1);
  // 先将第一个元素和已经排好的元素前一位做交换，再重新调整（刚调整的元素之前的元素），直到排序完毕
  for (int i = len - 1; i > 0; i--) {
    swap(arr[0], arr[i]);
    sift_down(arr, 0, i - 1);
  }
}
```

### 归并排序

```c++
// C++ version
void merge(int l, int r) {
  if (r - l <= 1) return;
  int mid = l + ((r - l) >> 1);
  merge(l, mid), merge(mid, r);
  for (int i = l, j = mid, k = l; k < r; ++k) {
    if (j == r || (i < mid && a[i] <= a[j]))
      tmp[k] = a[i++];
    else
      tmp[k] = a[j++];
  }
  for (int i = l; i < r; ++i) a[i] = tmp[i];
}
```

### 快速排序

```c++
void quick_sort(int a[],int s,int t)
{
    if(s==t)return;
    //选择基准位置 一般为第一个
    int tag=s;
    int i=s,j=t;//两个游标
    while(i<j)
    {
        while(a[j]>a[tag]&&j>tag)
        {
            j--;
        }
        swap(a[j],a[tag]);
        tag=j;
        
        
        while(a[i]<=a[tag]&&i<tag)
        {
            i++;
        }
        swap(a[i],a[tag]);
        tag=i;
    }
    if(s<=tag-1)quick_sort(a,s,tag-1);
    if(tag+1<=t)quick_sort(a,tag+1,t);

}
```

### 基数排序

```c++
const int N = 100010;
const int W = 100010;
const int K = 100;

int n, w[K], k, cnt[W];

struct Element {
  int key[K];

  bool operator<(const Element& y) const {
    // 两个元素的比较流程
    for (int i = 1; i <= k; ++i) {
      if (key[i] == y.key[i]) continue;
      return key[i] < y.key[i];
    }
    return false;
  }
} a[N], b[N];

void counting_sort(int p) {
  memset(cnt, 0, sizeof(cnt));
  for (int i = 1; i <= n; ++i) ++cnt[a[i].key[p]];
  for (int i = 1; i <= w[p]; ++i) cnt[i] += cnt[i - 1];
  // 为保证排序的稳定性，此处循环i应从n到1
  // 即当两元素关键字的值相同时，原先排在后面的元素在排序后仍应排在后面
  for (int i = n; i >= 1; --i) b[cnt[a[i].key[p]]--] = a[i];
  memcpy(a, b, sizeof(a));
}

void radix_sort() {
  for (int i = k; i >= 1; --i) {
    // 借助计数排序完成对关键字的排序
    counting_sort(i);
  }
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

### 线段树

```c++
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const ll N=100100;
ll n,m,a[N];
struct tree{
	ll l,r;
	ll pre,add;
}t[4*N];
void build_tree(ll p,ll l,ll r){
	t[p].l=l,t[p].r=r;
	if( l==r ) {
		scanf("&lld",&t[p].pre);
		return ;
	}
	ll mid=(l+r)/2;
	build_tree(p*2,l,mid);
	build_tree(p*2+1,mid+1,r);
	t[p].pre=t[p*2].pre+t[p*2+1].pre;
}
void lazy_tag(ll p){
	if( t[p].add ) {
		t[p*2].pre+=t[p].add*(t[p*2].r-t[p*2].l+1);
		t[p*2+1].pre+=t[p].add*(t[p*2+1].r-t[p*2+1].l+1);
		t[p*2].add+=t[p].add;
		t[p*2+1].add+=t[p].add;
		t[p].add=0;
	}
}
void update(ll p,ll x,ll y,ll z){
	if( x<=t[p].l && y>=t[p].r ) {
		t[p].pre+=(ll)z*(t[p].r-t[p].l+1);
		t[p].add+=z;
		return;
	}
	lazy_tag(p);
	ll mid=(t[p].l+t[p].r)/2;
	if( x<=mid ) update(p*2,x,y,z);
	if( y>mid ) update(p*2+1,x,y,z);
	t[p].pre=t[p*2].pre+t[p*2+1].pre;
}
ll query(ll p,ll x,ll y){
	if( x<=t[p].l && y>=t[p].r ) return t[p].pre;
	lazy_tag(p);
	ll mid=(t[p].l+t[p].r)/2;
	ll ans=0;
	if( x<=mid ) ans+=query(p*2,x,y);
	if( y>mid ) ans+=query(p*2+1,x,y);
	return ans;
}
int main(){
	scanf("%lld%lld",&n,&m);
	build_tree(1,1,n);
	for(ll i=1;i<=m;i++){
		ll p,x,y,k;
		scanf("%lld",&p);
		if( p==1 ){
			scanf("%lld%lld%lld",&x,&y,&k);
			update(1,x,y,k);
		} 
		if( p==2 ){
			scanf("%lld%lld",&x,&y);
			printf("%lld\n",query(1,x,y));
		} 
	}
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
        pair<int,int> temp = q.top();//记录堆顶，即堆内最小的边并将其弹出 
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

#### 树剖

```c++

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



