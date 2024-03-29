# ACM常用模板整理

[toc]

<div STYLE="page-break-after: always;"></div>

## 文件头

```c++
#pragma GCC optimize(2)
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#define int long long
#define cin std::cin
#define cout std::cout
#define fastio ios::sync_with_stdio(0), cin.tie(nullptr)
using namespace std;
const int N = 1e5 + 10;
const int mod = 998244353;
const int inf = 0x3fffffffffffffff;
char buf[1<<21],*p1=buf,*p2=buf;
inline char getc(){
    return p1==p2&&(p2=(p1=buf)+fread(buf,1,1<<21,stdin),p1==p2)?EOF:*p1++;
}
inline int read(){
    int ret = 0,f = 0;char ch = getc();
    while (!isdigit (ch)){
        if (ch == '-') f = 1;
        ch = getc();
    }
    while (isdigit (ch)){
        ret = ret * 10 + ch - 48;
        ch = getc();
    }
    return f?-ret:ret;
}
inline void solve() {

}
signed main() {
    fastio;
    int T;
    cin >> T;
    // T = 1;
    while(T --) {
        solve();
    }
    return 0;
}
```

## 读入输出

### __int128读入读出

```c++
#define int __int128
int read(){
    int ans=0,f=1;char c=getchar();
    while(!isdigit(c)){if(c=='-')f=-1;c=getchar();}
    while(isdigit(c)){ans=ans*10+c-'0';c=getchar();}
    return ans*f;
}
```

### 快读

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

### fread快读

```c++
char buf[1<<21],*p1=buf,*p2=buf;
inline char getc(){
    return p1==p2&&(p2=(p1=buf)+fread(buf,1,1<<21,stdin),p1==p2)?EOF:*p1++;
}
inline int read(){
    int ret = 0,f = 0;char ch = getc();
    while (!isdigit (ch)){
        if (ch == '-') f = 1;
        ch = getc();
    }
    while (isdigit (ch)){
        ret = ret * 10 + ch - 48;
        ch = getc();
    }
    return f?-ret:ret;
}
```

## 基础

### 位运算

```c++
// 获取 a 的第 b 位，最低位编号为 0
int getBit(int a, int b) { return (a >> b) & 1; }

// 将 a 的第 b 位设置为 0 ，最低位编号为 0
int unsetBit(int a, int b) { return a & ~(1 << b); }

// 将 a 的第 b 位设置为 1 ，最低位编号为 0
int setBit(int a, int b) { return a | (1 << b); }

// 将 a 的第 b 位取反 ，最低位编号为 0
int flapBit(int a, int b) { return a ^ (1 << b); }

// 求汉明权重，即它的 1 的个数
// 求 x 的汉明权重
int popcount1(int x) {
    int cnt = 0;
    while (x) {
        cnt += x & 1;
        x >>= 1;
    }
    return cnt;
}

// 求 x 的汉明权重
// 将这个数不断减去它的lowbit，直到这个数变为 0
int popcount2(int x) {
    int cnt = 0;
    while (x) {
        cnt++;
        x -= x & -x;
    }
    return cnt;
}

// 得到与数 x 汉明权重相等的后继; 注：0 需要特判， 0 无相同汉明权重的后继
void hj(int x){
    int t = x + (x & -x);
    x = t | ((((t&-t)/(x&-x))>>1)-1);
}
// 枚举 0 ~ n 按汉明权重递增的排列的完整代码为：
void hmqz1(int n){
    for (int i = 0; (1<<i)-1 <= n; i++) {
        for (int x = (1<<i)-1, t; x <= n; t = x+(x&-x), x = x ? (t|((((t&-t)/(x&-x))>>1)-1)) :(n+1)) {
        // 写下需要完成的操作
        }
    }
}

// 内建函数，运行速度更快

// 返回 x 的二进制末尾最后一个 1 的位置，位置的编号从 1 开始（最低位编号为 1 ）。当 x 为 0 时返回 0 。
// int __builtin_ffs(int x);

// 返回 x 的二进制的前导 0 的个数。当 x 为 0 时，结果未定义。
// int __builtin_clz(unsigned int x);

// 返回 x 的二进制末尾连续 0 的个数。当 x 为 0 时，结果未定义。
// int __builtin_ctz(unsigned int x);

// 当 x 的符号位为 0 时返回 x 的二进制的前导 0 的个数减一，否则返回 x 的二进制的前导 1 的个数减一
// int __builtin_clrsb(int x);

// 返回 x 的二进制中 1 的个数
// int __builtin_popcount(unsigned int x);

// 判断 x 的二进制中 1 的个数的奇偶性
// int __builtin_parity(unsigned int x);

// 求数 n 的以 2 为底的对数
// 求一个数以2 为底的对数相当于这个数的二进制的位数 -1 （不考虑0）
int ds(int n){
    return 31 - __builtin_clz(n);
}

int main(){
    int a = 31;
    cout << ds(a) << endl;
    return 0;
}
```

## DP

### 背包DP

```c++
// luogu P1757 
// 分组背包

#include<bits/stdc++.h>
#define int long long
using namespace std;

const int N = 1e3 + 10;
const int M = 1e6 + 10;

int dp[N];
int n, m;
int wo, vo, co;
vector<int> w[N], v[N];
int cnt = 0;
bool vis[N];

inline void solve() {
    scanf("%lld%lld", &m, &n);
    for(int i = 1; i <= n; ++i) {
        scanf("%lld%lld%lld", &wo, &vo, &co);
        if(!vis[co]) {
            cnt ++;
            vis[co] = 1;
        }
        w[co].push_back(wo);
        v[co].push_back(vo);
    }
    for(int i = 1; i <= cnt; ++i) {
        for(int j = m; j >= 1; --j) {
            for(int k = 0; k < w[i].size(); ++k) {
                if(j >= w[i][k]) {
                    dp[j] = max(dp[j], dp[j - w[i][k]] + v[i][k]);
                }
            }
        }
    }
    printf("%d\n", dp[m]);
}

signed main() {
    // ios::sync_with_stdio(0);
    // cin.tie(0), cout.tie(0);

    int T;
    T = 1;
    while(T--) {
        solve();
    }
    return 0;
}
```



### 最长公共子序列(LCS)

```c++
#include<bits/stdc++.h>
using namespace std;
int n;
int a1[100010],a2[100010];
int belong[100010];
int f[100010],b[100010],len;
int main()
{
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
    {
        scanf("%d",&a1[i]);
        belong[a1[i]]=i;
    }
    for(int i=1;i<=n;i++)
    scanf("%d",&a2[i]);
    for(int i=1;i<=n;i++)
    {
        if(belong[a2[i]]>b[len])
        {
            b[++len]=belong[a2[i]];
            f[i]=len;
            continue;
        }
        int k=lower_bound(b+1,b+len+1,belong[a2[i]])-b;
        b[k]=belong[a2[i]];
        f[i]=k;
    }
    printf("%d\n",len);
    return 0;
}
```

### 最长上升子序列

```c++
int cnt;
int mp[N];
int dp[N];//最长上升子序列初始化为inf，非上升为0
int  zcss()
{   
    for(int i=0;i<=cnt+1;i++) dp[i]=1e8;
    t=0;

    for(int i=0;i<cnt;i++)
    {
        int l=0,r=t+1;
        while((r-l)>1)
        {
            int m=(l+r)/2;
            if(dp[m]<mp[i]) l=m;
            else r=m;
        }

        int x=l+1;
        if(x>t) t=x;
        dp[x]=min(mp[i],dp[x]);
    }
    return t;
}
```

### 状压dp

状态压缩动态规划，就是我们俗称的状压DP，是利用计算机二进制的性质来描述状态的一种DP方式。

例题：在 n×n(1<=n<=10) 的棋盘上放 k(0<=k<n×n)个国王，国王可攻击相邻的 8 个格子，求使它们无法互相攻击的方案总数。

```c++
#include<bits/stdc++.h>
#define int long long
using namespace std;
const int maxn=160;
int f[11][maxn][maxn];
int num[maxn],s[maxn];
int n,k,cnt;
void init(){  //预处理一下没有其他行限制下每一行的可能情况有多少种
	cnt=0;
	for(int i=0;i<(1<<n);i++){
		if(i&(i<<1)){   // 代表左右有相邻国王
			continue;
		}
		int sum=0;
		for(int j=0;j<n;j++){  //枚举一下i这个情况下哪些地方是国王
			if(i&(1<<j)){
				sum++;
			}
		}
		s[++cnt]=i;  //s[cnt]代表了第cnt种情况下的状态  
		num[cnt]=sum;
	}
//	cout<<"cnt "<<cnt<<"\n";
}
void solve(){
	cin>>n>>k;
	init();
	f[0][1][0]=1;  //代表第0行在num[1]即放了0个国王的情况有1种
	for(int i=1;i<=n;i++){  //枚举行
		for(int j=1;j<=cnt;j++){  //枚举这一行有多少种情况
			for(int l=0;l<=k;l++){   //枚举算上这一行的国王总数
				if(l>=num[j]){  //算上这一行放的国王总数起码得大于等于这一行自己就有的国王个数
					for(int t=1;t<=cnt;t++){  //枚举上一行的情况
						//1.不能跟上一行有列重合 2.不能刚好差一行 
						if(!(s[t]&s[j])&&!(s[t]&(s[j]<<1))&&!(s[t]&(s[j]>>1))){
							f[i][j][l]+=f[i-1][t][l-num[j]];
						}
					}
				}
			}
		}
	}
	int ans=0;
	for(int i=1;i<=cnt;i++){
		ans+=f[n][i][k];
	}
	cout<<ans<<"\n";
}
signed main(){
	int t;
	t=1;
	while(t--){
		solve();
	}
}
```

```c++
    int sit[2000],gs[2000];
    int cnt=0;
    int n,yong;
    long long f[10][2000][100]={0};
    void dfs(int he,int sum,int node)//预处理出每一个状态
    {
        if(node>=n)//如果已经处理完毕（注意是大于等于）
        {
            sit[++cnt]=he;
            gs[cnt]=sum;
            return;//新建一个状态
        }
        dfs(he,sum,node+1);//不用第node个
        dfs(he+(1<<node),sum+1,node+2);//用第node个，此时node要加2，及跳过下一个格子
    }
    int main()
    {
        scanf("%d%d",&n,&yong);
        dfs(0,0,0);
        for(int i=1;i<=cnt;i++)f[1][i][gs[i]]=1;//第一层的所有状态均是有1种情况的
        for(int i=2;i<=n;i++)
            for(int j=1;j<=cnt;j++)
                for(int k=1;k<=cnt;k++)//枚举i、j、k
                {
                    if(sit[j]&sit[k])continue;
                    if((sit[j]<<1)&sit[k])continue;
                    if(sit[j]&(sit[k]<<1))continue;//排除不合法国王情况
                    for(int s=yong;s>=gs[j];s--)f[i][j][s]+=f[i-1][k][s-gs[j]];//枚举s，计算f[i][j][s]
                }
        long long ans=0;
        for(int i=1;i<=cnt;i++)ans+=f[n][i][yong];//统计最终答案，记得用long long
        printf("%lld",ans);
        return 0;
}
```

### 区间DP

```c++
for(int len = 1;len<=n;len++){//枚举长度
        for(int j = 1;j+len<=n+1;j++){//枚举起点，ends<=n
            int ends = j+len - 1;
            for(int i = j;i<ends;i++){//枚举分割点，更新小区间最优解
                dp[j][ends] = min(dp[j][ends],dp[j][i]+dp[i+1][ends]+something);
            }
        }
}
```

朴素区间dp（n^3)

N堆石子摆成一条线。现要将石子有次序地合并成一堆。规定每次只能选相邻的2堆石子合并成新的一堆，并将新的一堆石子数记为该次合并的代价。计算将N堆石子合并成一堆的最小代价。

```c++
dp[j][ends] = min(dp[j][ends],dp[j][i]+dp[i+1][ends]+weigth[i][ends]);
```

### 数位DP

```c++
#include<iostream>
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define maxn 15
int num;
int* a = new int[maxn];
int f[15];
//int a[maxn];
int b[maxn];//b保存第p为存的是那个数
int ten[maxn];
int L, R;
int t;
int dfs(int p, bool limit) {//p表示在第p位，limite表示此时是否处于限制位
	if (p < 0) {
		//for (int i = 2; i >= 0; i--)cout << b[i];//无限递归,记得加结束return
		//cout << endl;
		return 0;//搜索结束，返回
	}
	if (!limit && f[p] != -1) {//记忆化搜索，不处于限制位，并且f[p]被算过了
		return f[p];
	}
	int up = limit ? a[p] : 9;//判断是否处于限制位，如果是就只能取到a[p]为，否则最高位能取到9
 
	int ans = 0;
 
	for (int i = 0; i <= up; i++) {
		//b[p] = i;
		if (i == 3) {
			if (limit && i == up) {
				ans += 1;
				for (int j = p - 1; j >= 0; j--)//处于限制条件就把限制数下面全算上
					ans += a[j] * ten[j];
			}
			else//如果不处于限制条件直接加上10的p次方
				ans += ten[p];
		}
		else ans += dfs(p - 1, limit && i == up);//这里填a[p]可以填up也行，在处于限制的时候up等于a[p]
 
	}
	if (!limit)//记忆化搜索，如果没有处于限制条件就可以直接那算过一次的数直接用，能节省很多时间
		f[p] = ans;
	return ans;
}
 
int handle(int num) {
	int p = 0;
	while (num) {//把num中的每一位放入数组
		a[p++] = num % 10;
		num /= 10;
	}
	//说明a数组写进去了，但是读取无效数据是什么意思勒，之前好像不是这样的，解决办法，动态创建数组
	/*for (int i = 0; i < p; i++) {
		cout << a[i];
	}*/
	return dfs(p - 1, true);//对应的最高位为p-1位，为True表示没有处于限制位
}
 
void init() {
	ten[0] = 1;
	for (int i = 1; i < 15; i++) {
		ten[i] = ten[i - 1] * 10;
	}
	memset(f, -1, sizeof(f));
}
int32_t  main() {
	cin>>t;
    while(t--){
        cin>>L>>R;
        //handle(23);
	    init();//一定要记得初始化，TM的我在这卡了半个月
	    cout << handle(R)-handle(L) << endl;
	    delete[]a;
    }
    return 0;
}
```

### 概率DP

顾名思义，概率DP就是动态规划求概率的问题。一般来说，我们将dp数组存放的数据定义为到达此状态的概率，那么我们初值设置就是所有初始状态概率为1，最终答案就是终末状态dp值了。

我们在进行状态转移时，是从初始状态向终末状态顺推，转移方程中大致思路是按照当前状态去往不同状态的位置概率转移更新DP，且大部分是加法。

### 记忆化DP

题目连接：[花店橱窗](https://ac.nowcoder.com/acm/contest/24213/1005)

```c++
#pragma GCC optimize(2)
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#define int long long
#define cin std::cin
#define cout std::cout
#define fastio ios::sync_with_stdio(0), cin.tie(nullptr)
using namespace std;
const int N = 1e5 + 10;
const int mod = 1e9 + 7;
const int inf = 0x3fffffffffffffff;
char buf[1<<21],*p1=buf,*p2=buf;
inline char getc(){
    return p1==p2&&(p2=(p1=buf)+fread(buf,1,1<<21,stdin),p1==p2)?EOF:*p1++;
}
inline int read(){
    int ret = 0,f = 0;char ch = getc();
    while (!isdigit (ch)){
        if (ch == '-') f = 1;
        ch = getc();
    }
    while (isdigit (ch)){
        ret = ret * 10 + ch - 48;
        ch = getc();
    }
    return f?-ret:ret;
}
int n, f, v;
int dp[105][105], a[105][105];
vector<int> ans[105];

// 应用答案回溯
void out_ans(int x, int y) {
    if (!x) return;
    if (dp[x][y] == dp[x][y - 1]) {
        out_ans(x, y - 1);
    } else {
        out_ans(x - 1, y - 1);
        cout << y << ' ';
    }
}
inline void solve() {
    cin >> f >> v;
    for(int i = 1; i <= f; ++i) {
        for(int j = 1; j <= v; ++j) {
            cin >> a[i][j];
        }
        dp[i][i - 1] = -inf;
    }
    for(int i = 1; i <= f; ++i) {
        for(int j = i; j <= v; ++j) {
            dp[i][j] = max(dp[i][j - 1], dp[i - 1][j - 1] + a[i][j]);
        }
    }
    cout << dp[f][v] << endl;
    out_ans(f, v);
    return ;

}
signed main() {
    fastio;
    int T;
    // cin >> T;
    T = 1;
    while(T --) {
        memset(dp, 0, sizeof(dp));
        solve();
    }
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

### vector用法

#### 1.初始化：

```c++
vector<类型>标识符
vector<类型>标识符(最大容量)
vector<类型>标识符(最大容量,初始所有值)
 
int i[5]={1,2,3,4,5}
vector<类型>vi(i,i+2);//得到i索引值为3以后的值
 
vector<vector<int>>v; 二维向量//这里最外的<>要有空格。否则在比较旧的编译器下无法通过
```

#### 2.常用函数

```c++
push_back()  //在数组的最后添加一个数据
 
pop_back() //去掉数组的最后一个数据

at()  //得到编号位置的数据

begin() //得到数组头的指针

end() //得到数组的最后一个单元+1的指针
  
find()  //判断元素是否存在

front() //得到数组头的引用

back() //得到数组的最后一个单元的引用

max_size() //得到vector最大可以是多大

capacity() //当前vector分配的大小

size() //当前使用数据的大小 or 返回a在内存中总共可以容纳的元素个数

a.reserve(100); //改变当前vecotr所分配空间的大小将a的容量（capacity）扩充至100，也就是说现在测试a.capacity();的时候返回值是100

a.resize(10); //将a的现有元素个数调至10个，多则删，少则补，增加的元素其值默认为0

a.resize(10,2); //将a的现有元素个数调至10个，多则删，少则补，增加的元素其值为2

erase() //删除指针指向的数据项

clear() //清空当前的vector

rbegin() //将vector反转后的开始指针返回(其实就是原来的end-1)

rend() //将vector反转构的结束指针返回(其实就是原来的begin-1)

empty() //判断vector是否为空

swap() //与另一个vector交换数据
a.swap(b); //b为向量，将a中的元素和b中的元素进行整体性交换

reverse(obj.begin(),obj.end());反向迭代器,实现元素对调
```

#### 3.find用法

```c++
find(数组的头地址, 数组的尾地址, 要找的数)
 
find(nums.begin(), nums.end(), target)

//返回的是target第一次出现的地址
//如果没有找到返回尾地址nums.end()
```

**实例**

```c++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
using std::vector;
using std::cout;
using std::endl;
int main() {
    vector<int> nums = {2,7,8,8,9};
    int target = 8;
    vector<int>::iterator loc = find(nums.begin(), nums.end(), target);

    if (loc == nums.end()) {
        cout << "数组中没有" << target << endl;
    }
    else {
        cout << "数组中有" << target << endl;
        cout << "并且, 它的第一次出现的位置为:" << loc - nums.begin() << endl;
    }
}
```

#### 4.访问

```c++
#include <bits/stdc++.h>
using namespace std;
int main()
{
    //顺序访问
    vector<int>obj;
    for(int i=0;i<10;i++)
    {
        obj.push_back(i);//存储数据
    }
    
    //方法一数组访问
    cout<<"直接利用数组：";
    for(int i=0;i<10;i++)
    {
        cout<<obj[i]<<" ";
    }
    cout<<endl;
    
  
    //方法二，使用迭代器将容器中数据输出
    cout<<"利用迭代器：" ;
    vector<int>::iterator it;
    //声明一个迭代器，来访问vector容器，作用：遍历或者指向vector容器的元素
    for(it=obj.begin();it!=obj.end();it++)
    {
        cout<<*it<<" ";
    }
  
    return 0;
}
```

#### 5.insert插入

```c++
insert()  //往vector任意位置插入一个元素，指定位置或者指定区间进行插入，
 //第一个参数是个迭代器,第二个参数是元素。返回值是指向新元素的迭代器

vector<int> vA;
vector<int>::iterator it;

//指定位置插入
//iterator insert(const_iterator _Where, const _Ty& _Val)
//第一个参数是个迭代器位置,第二个参数是元素
it = vA.insert(vA.begin(),2); //往begin()之前插入一个int元素2 (vA={2,1}) 此时*it=2

  
//指定位置插入
//void insert(const_iterator _Where, size_type _Count, const _Ty& _Val) 
//第一个参数是个迭代器位置,第二个参数是要插入的元素个数，第三个参数是元素值
it = vA.insert(vA.end(),2,3);//往end()之前插入2个int元素3 (vA={2,1,3,3}) 此时*it=3


//指定区间插入
//void insert(const_iterator _Where, _Iter _First, _Iter _Last) 
vector<int> vB(3,6);  //vector<类型>标识符(最大容量,初始所有值)
it = vA.insert(vA.end(),vB.begin(),vB.end()); //把vB中所有元素插入到vA的end()之前 (vA={2,1,3,3,6,6,6})
//此时*it=6,指向最后一个元素值为6的元素位置



//删除元素操作：
 
pop_back()  从vector末尾删除一个元素

erase()  从vector任意位置删除一个元素，指定位置或者指定区间进行删除，第一个参数都是个迭代器。返回值是指向删除后的下一个元素的迭代器

clear()   清除vector中所有元素, size=0, 不会改变原有capacity
```

#### 6.排序

```c++
sort(v.begin(), v.end(),less<int>()); // 升序
sort(v.begin(), v.end(),greater<int>()); // 降序
```

#### 7.删除元素

+ $erase()$ 函数：

  $erase()$ 函数用于在顺序型容器中删除容器的一个元素，有两种函数原型，$c.erase (p ),c.erase(b,e);$  第一个删除迭代器 $p$ 所指向的元素，第二个删除迭代器 $b,e$ 所标记的范围内的元素，$c$ 为容器对象，返回值都是一个迭代器，该迭代器指向被删除元素后面的元素（这个是重点）

```c++
for(auto iter=vec.begin();iter!=vec.end(); ) {
    if( *iter == 3) iter = veci.erase(iter);//当删除时erase函数自动指向下一个位置，就不需要进行++
    else iter ++ ;    //当没有进行删除的时候，迭代器++
}
```

+ $remove()$ 函数：

  $remove$ 是个stl的通用算法 `std::remove(first,last,val)` 移除 $[first, last)$ 范围内等于 $val$ 的元素在$vector$ 里面用就类似于 `iter=std::remove(vec.begin(), vec.end(), val)` 但这个函数只是把 $val$ 移到$vec$ 的末尾，并不真正删除,真正删除还是要调用一次erase函数

```c++
veci.erase(remove(vec.begin(),vec.end(),3),vec.end());
```

+ 重复元素

### bitset用法

bitset可以说是一个多位二进制数，每八位占用一个字节，因为支持基本的位运算，所以可用于状态压缩，n位bitset执行一次位运算的时间复杂度可视为n/32.

输出只能用cout

#### 1.构造：

```c++
int a = 5;
string b = "1011";
char c[4] = {'1','0','1','0'};
bitset<10>s1(string("1001"));     //0000001001
bitset<10>s2(int(8));             //0000001000
bitset<10>s3(8);                  //0000001000
bitset<4>s4(string("10001"));     //1000
bitset<4>s5(int(32));             //0000
bitset<4>s6;                      //0000
bitset<4>s7(a);                   //0101
bitset<4>s8(b);                   //1011
bitset<4>s9(c);                   //1010
```

+ 不够的位数自动补0
+ size小于附的值时，int取后几位，string取前几位

+ 不进行赋初值时，默认全部为0

#### 2.运算：

```c++
bitset<4>s1(string("1001"));
bitset<4>s2(string("1000"));
s1[1] = 1;                    
cout<<s1[0]<<endl;              //1
cout<<s1<<endl;                 //1011
cout<<(s1==s2)<<endl;           //0
cout<<(s1!=s2)<<endl;           //1
cout<<(s1^s2)<<endl;            //0011
cout<<(s1&s2)<<endl;            //1000
cout<<(s1|s2)<<endl;            //1011
cout<<(~s1)<<endl;              //0100
cout<<(s1>>1)<<endl;            //0101
```

#### 3.函数：

```c++
bitset<4>s1(string("1001"));
bitset<4>s2(string("0011"));

cout<<s1.count()<<endl;//用于计算s1中1的个数
cout<<s1.size()<<endl;//s1的位数

cout<<s1.test(0)<<endl;//用于检查s1[0]是0or1并返回0or1
cout<<s1.any()<<endl;//检查s1中是否有1，并返回1or0
cout<<s1.all()<<endl;//检查s1中是否全部为1，并返回0or1
cout<<s1.none()<<endl;//检查s1中是否全不为1，并返回0or1

// flip
cout<<s1.flip(2)<<endl;//将参数位取反，0变1，1变0
cout<<s1.flip()<<endl;//不指定参数时，每一位取反

// set
cout<<s1.set()<<endl;//不指定参数时，每一位变为１
cout<<s1.set(3,1)<<endl;//指定两位参数时，将第一参数位的元素变为第二参数的值，第二参数位只能为0or1
cout<<s1.set(3)<<endl;//只有一个参数时，将参数下标处变为１

// reset
cout<<s1.reset(4)<<endl;//一个参数时将参数下标处变为０
cout<<s1.reset()<<endl;//不传参数时将bitset的每一位变为０

string s = s1.to_string();　　//将bitset转换成string
unsigned long a = s1.to_ulong();　　//将bitset转换成unsigned long
unsigned long long b = s1.to_ullong();　　//将bitset转换成unsigned long long
```

### 并查集（路经压缩）

```c++
int fa[N];
inline void init() { for(int i = 1; i <= n; ++i) fa[i] = i; }
int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
void merge(int x, int y) { fa[find(x)] = find(y); }
```

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
#include <iostream>

using namespace std;

typedef long long LL;

const int N = 100010;

int n, m;  // 数列长度、操作个数
int a[N];  // 输入的数组
struct Node {
    int l, r;
    LL sum;  // 如果考虑当前节点及子节点上的所有标记，其区间[l, r]的总和就是sum
    LL add;  // 懒标记，表示需要给以当前节点为根的子树中的每一个节点都加上add这个数(不包含当前节点)
} tr[N * 4];

// 由子节点的信息，来计算父节点的信息
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}

// 把当前父节点的修改信息下传到子节点，也被称为懒标记（延迟标记）
void pushdown(int u) {
    
    auto &root = tr[u], &left = tr[u << 1], &right = tr[u << 1 | 1];
    if (root.add) {
        left.add += root.add, left.sum += (LL)(left.r - left.l + 1) * root.add;
        right.add += root.add, right.sum += (LL)(right.r - right.l + 1) * root.add;
        root.add = 0;
    }
}

// 创建线段树
void build(int u, int l, int r) {
    
    if (l == r) tr[u] = {l, r, a[l], 0};
    else {
        tr[u] = {l, r};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

// 将a[l~r]都加上d
void modify(int u, int l, int r, LL d) {
    
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum += (LL)(tr[u].r - tr[u].l + 1) * d;
        tr[u].add += d;
    } else {  // 一定要分裂
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid) modify(u << 1, l, r, d);
        if (r > mid) modify(u << 1 | 1, l, r, d);
        pushup(u);
    }
}

// 返回a[l~r]元素之和
LL query(int u, int l, int r) {
    
    if (tr[u].l >= l && tr[u].r <= r) return tr[u].sum;
    
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    LL sum = 0;
    if (l <= mid) sum += query(u << 1, l, r);
    if (r > mid) sum += query(u << 1 | 1, l, r);
    return sum;
}

int main() {
    
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
    
    build(1, 1, n);
    
    char op[2];
    int l, r, d;
    while (m--) {
        scanf("%s%d%d", op, &l, &r);
        if (*op == 'C') {
            scanf("%d", &d);
            modify(1, l, r, d);
        } else {
            printf("%lld\n", query(1, l, r));
        }
    }
    
    return 0;
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

### 倍增思想（ST表）

```c++
#include<bits/stdc++.h>
#define int long long 
using namespace std;
const int N = 1e5 + 10;
const int mod = 1e9 + 7;
int n, m, k;
int lg[N], st[N][30];

inline int read()
{
    int x=0,f=1;char ch=getchar();
    while (ch<'0'||ch>'9'){if (ch=='-') f=-1;ch=getchar();}
    while (ch>='0'&&ch<='9'){x=x*10+ch-48;ch=getchar();}
    return x*f;
}

inline void print(int x)
{
    if(x==0)return;
    print(x/10);
    putchar(x%10+'0');
}

signed main() {
    ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
    n = read(), m = read();
    for(int i = 1; i <= n; ++i) {
        st[i][0] = read();
        if(i >= 2) lg[i] = lg[i >> 1] + 1;
    }
    for(int j = 1; j <= lg[n] + 1; ++j) {
        for(int i = 1; i + (1 << j) - 1 <= n; ++i) {
            st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
        }
    }
    while(m --) {
        int l, r;
        l = read(), r = read();
        int mid = lg[r - l + 1];
        print(max(st[l][mid], st[r - (1 << mid) + 1][mid]));
        printf("\n");
        // std::cout << max(st[l][mid], st[r - (1 << mid) + 1][mid]) << endl;
    }
    return 0;
}
```

### 倍增思想（LCA）

```c++
#pragma GCC optimize(2)
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#define int long long
#define cin std::cin
#define cout std::cout
#define fastio ios::sync_with_stdio(0), cin.tie(nullptr)
using namespace std;
const int N = 4e4 + 10;
const int mod = 998244353;
const int inf = 0x3fffffffffffffff;
char buf[1<<21],*p1=buf,*p2=buf;
inline char getc(){
    return p1==p2&&(p2=(p1=buf)+fread(buf,1,1<<21,stdin),p1==p2)?EOF:*p1++;
}
inline int read(){
    int ret = 0,f = 0;char ch = getc();
    while (!isdigit (ch)){
        if (ch == '-') f = 1;
        ch = getc();
    }
    while (isdigit (ch)){
        ret = ret * 10 + ch - 48;
        ch = getc();
    }
    return f?-ret:ret;
}
std::vector<int> G[N], W[N];
int n, m;
int fa[N][31], dep[N], cost[N][31];
inline void init() {
	for(int i = 1; i < N; ++i) {
		G[i].clear(), W[i].clear();
	}
	memset(fa, 0, sizeof(fa));
	memset(dep, 0, sizeof(dep));
	memset(cost, 0, sizeof(cost));
}
inline void dfs(int u, int fno) {
    // 初始化：第 2^0 = 1 个祖先就是它的父亲节点，dep 也比父亲节点多 1。
    fa[u][0] = fno;
    dep[u] = dep[fno] + 1;
    // 初始化：其他的祖先节点：第 2^i 的祖先节点是第 2^(i-1) 的祖先节点的第2^(i-1) 的祖先节点。
    for(int i = 1; i <= 30; ++i) {
        fa[u][i] = fa[fa[u][i - 1]][i - 1];
        cost[u][i] = cost[fa[u][i - 1]][i - 1] + cost[u][i - 1];
    }
    int sz = G[u].size();
    for(int i = 0; i < sz; ++i) {
        if(G[u][i] == fno) continue;
        cost[G[u][i]][0] = W[u][i];
        dfs(G[u][i], u);
    }
}
int lca(int x, int y) {
	if(dep[x] > dep[y]) swap(x, y);
	int tmp = dep[y] - dep[x], ans = 0;
	for(int j = 0; tmp; ++j, tmp >>= 1) {
		if (tmp & 1) ans += cost[y][j], y = fa[y][j];
	}
    
	if(y == x) return ans;
	for(int j = 30; j >= 0 && y != x; --j) {
		if (fa[x][j] != fa[y][j]) {
	        ans += cost[x][j] + cost[y][j];
	        x = fa[x][j];
	        y = fa[y][j];
	    }
	}
	ans += cost[x][0] + cost[y][0];
	return ans;

}
inline void solve() {
    cin >> n >> m;
    for(int i = 1; i < n; ++i) {
        int s, t, w;
        cin >> s >> t >> w;
        G[s].push_back(t);
        G[t].push_back(s);
        W[s].push_back(w);
        W[t].push_back(w);
    }
    dfs(1, 0);
    // cin >> m;
    for(int i = 1; i <= m; ++i) {
    	int x, y;
    	cin >> x >> y;
    	cout << lca(x, y) << endl;
    }
    return;

}
signed main() {
    fastio;
    int T;
    cin >> T;
    // T = 1;
    while(T --) {
    	init();
        solve();
    }
    return 0;
}
```

### 舞蹈链

### 主席树

### 红黑树

## 图论

### 树

#### LCA



### 最短路算法

#### Floyed

#### Dijkstra

```c++
void dijkstra(int n)
{
  	typedef pair<int, int> pii;
    priority_queue<pii,vector<pii>,greater<pii> >q;
    memset(dis,0x3f,sizeof dis);
    q.push(make_pair(0,t));
    dis[t]=0;
    while(!q.empty())
    {
        pii temp = q.top();//记录堆顶，即堆内最小的边并将其弹出 
        q.pop();
        int u = temp.second;//点 
        if(vis[u]) continue;//如果被访问过，就跳过 
        vis[u]=true;//标记 
        for(int i = head[u];i!=-1;i=edge[i].nxt)//搜索堆顶的所有连边 
        {
            int v = edge[i].to;
            if(dis[v]>dis[u]+edge[i].val)//松弛操作 
            {
                dis[v]=dis[u]+edge[i].val;
                q.push(make_pair(dis[v],v));//把新遍历的点加到堆中 
            }
        }
    }
    //for(int i=1;i<=n;i++) cout<<dis[i]<<endl;
}
```

#### SPFA

```c++
inline bool spfa(int S, int V)
{
	for (int i = 1; i <= V; ++i) {
		dis[i] = inf;
		cnt[i] = 0;
		vis[i] = 0;
	}
	vis[S] = 1; dis[S] = 0;
	q.push(S);
	while (!q.empty()) {
		int u = q.front(); q.pop(); vis[u] = 0;
		for (int i = head[u]; ~i; i = edge[i].nxt) {
			int to = edge[i].to;
			if (dis[to] > dis[u] + edge[i].val) {
				dis[to] = dis[u] + edge[i].val;
				if (!vis[to]) {
					if (++cnt[to] >= V)return false;
					vis[to] = 1; q.push(to);
				}
			}
		}
	}
	return true;
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

### Tarjan

#### 割点

```c++
// luogu P3388 【模板】割点（割顶）

#include <bits/stdc++.h>
using namespace std;
const int inf = 0x3f3f3f3f;
const int maxn = 2e4+10;
int n, m, t, root;
int ans = 0;

int head[maxn], ct;
struct Edge {
	int val, to, nxt;
}edge[maxn*10];

inline void addedge(int s, int t) {
	edge[++ct].to = t;
	edge[ct].nxt = head[s];
	head[s] = ct;
}

inline void init() {
	memset(head, -1, sizeof head);
	ct = 0;
}

int dfn[maxn], low[maxn], dfncnt, st[maxn], in_stack[maxn], tp;
int scc[maxn], sc;  // 结点 i 所在 SCC 的编号
int sz[maxn],ind[maxn],od[maxn];       // 强连通 i 的大小
int is_cut[maxn];

void tarjan(int u) {
	low[u] = dfn[u] = ++dfncnt;
	int cnt = 0;
	for (int i = head[u]; ~i; i = edge[i].nxt) {
    	int v = edge[i].to;
		if (!dfn[v]) {
			tarjan(v);
			low[u] = min(low[u], low[v]);
			if (low[v] >= dfn[u]) {
				cnt++;
				if (u != root || cnt > 1) {
					is_cut[u] = 1;
				}
			}
		}
		else low[u] = min(low[u], dfn[v]);
  	}
}

inline void solve() {
	cin >> n >> m;
	for (int i = 1; i <= m; ++i) {
		int s, t;
		cin >> s >> t;
		if (s == t)continue;
		addedge(s, t);
		addedge(t, s);
	}
	for (int i = 1; i <= n; ++i) {
		if (dfn[i] == 0) {
			root = i;
			tarjan(i);
		}
	}
	for (int i = 1; i <= n; ++i) {
		if (is_cut[i])ans++;
	}
	cout << ans << endl;
	for (int i = 1; i <= n; ++i) {
		if (is_cut[i])cout << i << " ";
	}
	return;
}

int main() {
	ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);
	int T;
	T = 1;
	while (T--) {
		init();
		solve();
	}
	return 0;
}
```

#### 缩点

```c++
int dfn[maxn], low[maxn], dfncnt, st[maxn], in_stack[maxn], tp;
int scc[maxn], sc;  // 结点 i 所在 SCC 的编号
int sz[maxn];       // 强连通 i 的大小


void tarjan(int u) {
  	low[u] = dfn[u] = ++dfncnt, st[++tp] = u, in_stack[u] = 1;
  	for (int i = head[u]; ~i; i = edge[i].nxt) {
    	int v = edge[i].to;
    	if (!dfn[v]) {
      		tarjan(v);
      		low[u] = min(low[u], low[v]);
    	} else if (in_stack[v]) {
      		low[u] = min(low[u], dfn[v]);
    	}
  	}
  	if (dfn[u] == low[u]) {
    	++sc;
    	while (st[tp] != u) {
      		scc[st[tp]] = sc;
      		sz[sc]++;
      		in_stack[st[tp]] = 0;
      		--tp;
    	}
    	scc[st[tp]] = sc;
    	sz[sc]++;
    	in_stack[st[tp]] = 0;
    	--tp;
  	}
}
```

### 二分图最大匹配

#### 匈牙利算法

+ 模板题：[luogu P3386 【模板】二分图最大匹配](https://www.luogu.com.cn/problem/P3386)
+ 时间复杂度：$O(n * e)$

```c++
#include<bits/stdc++.h>
#define int long long
#define cin std::cin
#define cout std::cout
#define fastio ios::sync_with_stdio(0), cin.tie(nullptr)
using namespace std;
const int N = 1e3 + 10;
const int mod = 998244353;
const int inf = 0x3fffffffffffffff;
char buf[1<<21],*p1=buf,*p2=buf;
inline char getc(){
    return p1==p2&&(p2=(p1=buf)+fread(buf,1,1<<21,stdin),p1==p2)?EOF:*p1++;
}
inline int read(){
    int ret = 0,f = 0;char ch = getc();
    while (!isdigit (ch)){
        if (ch == '-') f = 1;
        ch = getc();
    }
    while (isdigit (ch)){
        ret = ret * 10 + ch - 48;
        ch = getc();
    }
    return f?-ret:ret;
}
int n, m, t;
// mch存储右边到左边的匹配
int mch[N], vistime[N]; 
int mt[N];
std::vector<int> G[N];
inline bool dfs(int u, int tag) {
    if(vistime[u] == tag) return false;
    vistime[u] = tag;
    for(auto v : G[u]) {
        if(mch[v] == 0 || dfs(mch[v], tag)) {
            mch[v] = u;
            return true;
        }
    }
    return false;
}
signed main() {
    fastio;
    cin >> n >> m >> t;
    while(t--) {
        int u, v;
        cin >> u >> v;
        G[u].push_back(v);
    }
    int ans = 0;
    for(int i = 1; i <= n; ++i) {
        if(dfs(i, i)) ans ++;
    }
    // 处理出左边到右边的匹配
    for(int i = 1; i <= m; ++i) {
        if(mch[i]) mt[mch[i]] = i;
    }
    cout << ans;
    return 0;
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

### 线性组合数

```c++
int frac[N], inv[N];
// 快速幂
int qpow(int a, int b) {
    a %= mod;
    int s = 1;
    for (; b; a = 1ll * a * a % mod, b >>= 1) if (b & 1) s = 1ll * s * a % mod;
    return s;
}
// 线性求逆元
void set_up() {
    frac[0] = inv[0] = 1;
    for (int i = 1; i <= N - 1; i++) frac[i] = 1ll * frac[i - 1] * i % mod;
    inv[N - 1] = qpow(frac[N - 1], mod - 2);
    for (int i = N - 2; i; i--) inv[i] = 1ll * inv[i + 1] * (i + 1) % mod;
}
// O(1)计算组合数
inline int C(int n, int m) {
    if (n < m) return 0;
    return 1ll * frac[n] * inv[m] % mod * inv[n - m] % mod;
}
```

### 快速幂

```c++
// 取模版本
int qpow(int a, int b) {
    a %= mod;
    int s = 1;
    for (; b; a = 1ll * a * a % mod, b >>= 1) if (b & 1) s = 1ll * s * a % mod;
    return s;
}
// 求a的逆元
int inv = qpow(a, mod - 2);
// 不取模版本
int qpow(int a, int b) {
    int s = 1;
    for (; b; a = 1ll * a * a, b >>= 1) if (b & 1) s = 1ll * s * a;
    return s;
}
```

### gcd/lmp

```c++
int gcd(int a, int b){
    if(a == 0) return b;
    else if(b == 0) return a;
    return a % b ? gcd(b, a % b) : b;
}
int lmp(int a, int b){
   return a*b/gcd(a,b);
}
// 优化的更相减损术
ll stein(ll a, ll b){
    if(a < b) a ^= b, b ^= a, a ^= b;
    if(b == 0) return a;
    if((!(a & 1)) && (!(b & 1))) return stein(a >> 1, b >> 1) << 1;
    else if((a & 1) && (!(b & 1))) return stein(a, b >> 1);
    else if((!(a & 1)) && (b & 1)) return stein(a >> 1, b);
    else return stein(a - b, b);
}
```

### 欧拉筛

```c++
bool isprime[MAXN]; // isprime[i]表示i是不是素数
int prime[MAXN]; // 现在已经筛出的素数列表
int n; // 上限，即筛出<=n的素数
int cnt; // 已经筛出的素数个数

void euler()
{
    memset(isprime, true, sizeof(isprime)); // 先全部标记为素数
    isprime[1] = false; // 1不是素数
    for(int i = 2; i <= n; ++i) // i从2循环到n（外层循环）
    {
        if(isprime[i]) prime[++cnt] = i;
        // 如果i没有被前面的数筛掉，则i是素数
        for(int j = 1; j <= cnt && i * prime[j] <= n; ++j)
        // 筛掉i的素数倍，即i的prime[j]倍
        // j循环枚举现在已经筛出的素数（内层循环）
        {
            isprime[i * prime[j]] = false;
            // 倍数标记为合数，也就是i用prime[j]把i * prime[j]筛掉了
            if(i % prime[j] == 0) break;
            // 最神奇的一句话，如果i整除prime[j]，退出循环
            // 这样可以保证线性的时间复杂度
        }
    }
}
```

### 线性同余方程

```c++
int exgcd(int a, int b, int &x, int &y) {       // a为分母，b为模数，x为所求
    if (!b) {
        x = 1, y = 0;
        return a;
    }
    int  d = exgcd(b, a % b, x, y);
    int  z = x;
    x = y;
    y = z - a / b * y;
    return d;
}
int res = exgcd(mu, mod, x, y);
x = (x * zi / res % mod) % mod;
```

### 高精度

```c++
#include <cstdio>
#include <cstring>

static const int LEN = 1004;

int a[LEN], b[LEN], c[LEN], d[LEN];

void clear(int a[])
{
    for (int i = 0; i < LEN; ++i)
        a[i] = 0;
}

void read(int a[])
{
    static char s[LEN + 1];
    scanf("%s", s);

    clear(a);

    int len = strlen(s);
    for (int i = 0; i < len; ++i)
        a[len - i - 1] = s[i] - '0';
}

void print(int a[])
{
    int i;
    for (i = LEN - 1; i >= 1; --i)
        if (a[i] != 0)
            break;
    for (; i >= 0; --i)
        putchar(a[i] + '0');
    putchar('\n');
}

void add(int a[], int b[], int c[])
{
    clear(c);

    for (int i = 0; i < LEN - 1; ++i)
    {
        c[i] += a[i] + b[i];
        if (c[i] >= 10)
        {
            c[i + 1] += 1;
            c[i] -= 10;
        }
    }
}

void sub(int a[], int b[], int c[])
{
    clear(c);

    for (int i = 0; i < LEN - 1; ++i)
    {
        c[i] += a[i] - b[i];
        if (c[i] < 0)
        {
            c[i + 1] -= 1;
            c[i] += 10;
        }
    }
}

void mul(int a[], int b[], int c[])
{
    clear(c);

    for (int i = 0; i < LEN - 1; ++i)
    {
        for (int j = 0; j <= i; ++j)
            c[i] += a[j] * b[i - j];

        if (c[i] >= 10)
        {
            c[i + 1] += c[i] / 10;
            c[i] %= 10;
        }
    }
}

bool greater_eq(int a[], int b[], int last_dg, int len)
{
    if (a[last_dg + len] != 0)
        return true;
    for (int i = len - 1; i >= 0; --i)
    {
        if (a[last_dg + i] > b[i])
            return true;
        if (a[last_dg + i] < b[i])
            return false;
    }
    return true;
}

void div(int a[], int b[], int c[], int d[])
{
    clear(c);
    clear(d);

    int la, lb;
    for (la = LEN - 1; la > 0; --la)
        if (a[la - 1] != 0)
            break;
    for (lb = LEN - 1; lb > 0; --lb)
        if (b[lb - 1] != 0)
            break;
    if (lb == 0)
    {
        puts("> <");
        return;
    }

    for (int i = 0; i < la; ++i)
        d[i] = a[i];
    for (int i = la - lb; i >= 0; --i)
    {
        while (greater_eq(d, b, i, lb))
        {
            for (int j = 0; j < lb; ++j)
            {
                d[i + j] -= b[j];
                if (d[i + j] < 0)
                {
                    d[i + j + 1] -= 1;
                    d[i + j] += 10;
                }
            }
            c[i] += 1;
        }
    }
}

int main()
{
    read(a);

    char op[4];
    scanf("%s", op);

    read(b);

    switch (op[0])
    {
        case '+':
            add(a, b, c);
            print(c);
            break;
        case '-':
            sub(a, b, c);
            print(c);
            break;
        case '*':
            mul(a, b, c);
            print(c);
            break;
        case '/':
            div(a, b, c, d);
            print(c);
            print(d);
            break;
        default:
            puts("> <");
    }

    return 0;
}
```

### 卢卡斯定理

```c++
const int maxn=100010;
long long mul[maxn];
void init(long long p){
    mul[0] = 1;
    for(long long i = 1; i <= p; i++) mul[i] = mul[i-1] * i % p;
}
// 卢卡斯定理要求模数必须是 素数
long long quickpow(long long a,long long b,long long c){  // c 是模数
    long long ans=1;a=a%c;
    while(b)
    {
        if(b&1) ans=(ans*a)%c;
        b>>=1;a=(a*a)%c;
    }
    return ans;
}
long long c(long long n,long long m,long long p){  // p 是模数
    return (m>n)?0:((mul[n]*quickpow(mul[m],p-2,p))%p*quickpow(mul[n-m],p-2,p)%p);
}
long long lucas(long long n,long long m,long long p){
    return (m==0)?1:c(n%p,m%p,p)*lucas(n/p,m/p,p)%p;
}
```

## 字符串

### 字符串哈希

```c++
const unsigned long long Mod=212370440130137957ll;
const int prime=233317;
const int bas=131;
unsigned  long long a[10010];
char ss[10010];
unsigned long long gethash()
{
    int len=strlen(ss+1);
    unsigned long long tmp=0;
        for(int i=1;i<=len;i++)
        {
            tmp=((tmp*bas%Mod+(unsigned long long)ss[i])%Mod+prime)%Mod;
        }
    return tmp;
}
```

### KMP

```c++
const int MAXN=1000100;
char s[MAXN],t[MAXN];//s是主串；t是模式串
int m,n;//m为模式串长度；n为主串长度
int nxt[MAXN];
void kmp_pre() {
    int i, j;
    j = nxt[0] = -1;
    i = 0;
    while(i < m) {
        if(j == -1 || t[i] == t[j]) nxt[++i] = ++j;
        else j = nxt[j];
    }
}
//出现次数kmp
int kmp_count() {
    int i = 0, j = 0;
    int ans = 0;
    if(m == 1 && n == 1)
    {
        if(t[0] == s[0]) return 1;
        else return 0;
    }
    kmp_pre();
    for(i = 0; i < n; i++)
    {
        while(j > 0 && s[i] != t[j]) j = nxt[j];
        if(s[i] == t[j]) j ++;
        if(j == m)
        {
            ans ++;
            j = nxt[j];
        } 
    }
    return ans;
}
//返回位置kmp
int KMP() {
    kmp_pre();
    int i = 0;
    int j = 0;
    while(i < n && j < m) {
        if(j == -1 || t[i] == p[j]) {
            i ++;
            j ++;
            if(j == m && i != n){//当模式串到达结尾时，回到指定位置
                j = nxt[j];
            }
        } else {
           j = nxt[j];
        }
    }
    return j;//返回前缀的位置
}
```

### 扩展KMP

```c++
#include<bits/stdc++.h>
#define magic ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
using namespace std;
const int maxn=2e7+5;
char a[maxn],b[maxn];
int n,m;
long long z[maxn],ext[maxn];
void getz(char *s,int n)//s的每一个后缀与s的LCP
{
    long long l=0,r=0;
    z[1]=n;
    for(int i=2;i<=n;i++)
    {
        if(i>r)
        {
            while(s[i+z[i]]==s[z[i]+1]) z[i]++;//超出范围的情况直接暴力枚举
            l=i,r=i+z[i]-1;//暴力枚举必然会更新维护区间
        }
        else if(z[i-l+1]<r-i+1) //可以直接通过前面已经获得的信息得到
        {
            z[i]=z[i-l+1];
        }
        else //前方的前缀过长，可取一部分，然后继续暴力枚举
        {
            z[i]=r-i+1;
            while(s[i+z[i]]==s[z[i]+1]) z[i]++;
            l=i,r=i+z[i]-1;
        }
    }
}
void getext(char *s1,int n,char*s2,int m)//s1(文本串)的每一个后缀与s2(模式串)的LCP，结果即数组ext
{
    long long l=0,r=0;
    for(int i=1;i<=n;i++)
    {
        if(i>r)
        {
            while(i+ext[i]<=n&&ext[i]+1<=m&&s1[i+ext[i]]==s2[ext[i]+1])
            ext[i]++;
            l = i, r = i + ext[i] - 1;
        }
        else if(z[i-l+1]<r-i+1) ext[i] = z[i - l + 1];
        else
        {
            ext[i]=r-i+1;
            while(i+ext[i]<=n&&ext[i]+1<=m&&s2[ext[i]+1]==s1[i+ext[i]])
            ext[i]++;
            l=i,r=i+ext[i]-1;
        }
    }
}
```

### Manacher

```c++
#include<bits/stdc++.h>
using namespace std;
const int maxn=11000010;
char ss[maxn];
char ns[maxn<<2];
int  d[maxn<<2];
int l,r;
int res=-1;
void manacher()
{
    ns[0]='@';
    ns[1]='#';
    int k=1;
    int len=strlen(ss);
    for(int i=0;i<len;i++)
    {
        ns[++k]=ss[i];
        ns[++k]='#';
    };
    len=k;
    d[1]=1;
    for(int i=1,l=r=1;i<=len;i++)
    {
        if(i<r) d[i]=min(d[r-i+l],r-i+1);
        while(ns[i-d[i]]==ns[i+d[i]]) d[i]++;
        if(i+d[i]>r)
        {
            l=i-d[i]+1;
            r=i+d[i]-1;
        }
        res=max(d[i],res);
    }
    return ;
}
int main()
{
    scanf("%s",ss);
    manacher();
    printf("%d\n",res-1);
    return 0;
}
```

### AC自动机

```c++
模板一：
#include<bits/stdc++.h>//求解文本串中出现了多少个模式串
#define magic ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
using namespace std;
const int maxn=1e6+10;
int tri[maxn][26];
int fail[maxn];
int tot;
int vis[maxn];
void Trie(string ms)
{
    int p=0;
    int len=ms.length();
    for(int i=0;i<len;i++)
    {
        int c=ms[i]-'a';
        if(!tri[p][c]) tri[p][c]=++tot;
        p=tri[p][c];
    }
    vis[p]++;
    return ;
}
void buildac()
{
    queue<int> q;
    for(int i=0;i<26;i++)
    {
        if(tri[0][i]) q.push(tri[0][i]);
    }

    while(!q.empty())
    {
        int u=q.front();
        q.pop();

        for(int i=0;i<26;i++)
        {
            if(tri[u][i])
            {
                fail[tri[u][i]]=tri[fail[u]][i];
                q.push(tri[u][i]);
            }
            else tri[u][i]=tri[fail[u]][i];
        }
    }
    return ;
}
int query(string ts)
{
    int len=ts.length();
    int u=0,res=0;
    for(int i=0;i<len;i++)
    {
        u=tri[u][ts[i]-'a'];
        for(int j=u;j&&vis[j]!=-1;j=fail[j])
        {
            res+=vis[j];
            vis[j]=-1;
        }
    }
    return res;
}
int main()
{
    magic 
    int n;
    cin>>n;
    string ms;
    for(int i=1;i<=n;i++)
    {
        cin>>ms;
        Trie(ms);
    }

    buildac();

    string ts;
    cin>>ts;

    int res=query(ts);
    cout<<res<<'\n';
    return 0;
}


模板二：
#include<bits/stdc++.h>//可以计算在文本串中出现最多的模式串并输出这个模式串
using namespace std;
const int maxn1=50000;
const int maxn2=1000010;
char t[maxn2],s[160][maxn1];
int n,num[160],e[maxn2],val[maxn2];
int trie[maxn1][30],cnt,fail[maxn1];
void init()
{
	memset(fail,0,sizeof(fail));
	memset(num,0,sizeof(num));
	memset(e,0,sizeof(e));
	memset(trie,0,sizeof(trie));
	memset(val,0,sizeof(val));
	cnt=0;
}
void insert(int l,char *s ,int id)
{
	int p=0;
	
	for(int i=1;i<=l;i++)
	{
		int c=s[i]-'a';
		if(!trie[p][c]) trie[p][c]=++cnt;
		p=trie[p][c];
	}
	e[p]=id;//需要注意的是
	//这个地方要注意标记下这个模式串最后一次出现的索引
	//如果有重复的模式串出现的话，这个地方的e[p]会被覆盖掉
	//但是并没有什么问题，因为这个求得是出现次数最多的模式串
	//相同的模式串不需要进行重复输出。
}
void build()
{
	queue<int> q;
	
	for(int i=0;i<26;i++)
	{
		if(trie[0][i]) q.push(trie[0][i]);
	}
	
	while(!q.empty())
	{
		int p=q.front();
		q.pop();
		
		for(int i=0;i<26;i++)
		{
			if(trie[p][i])
			{
				fail[trie[p][i]]=trie[fail[p]][i];
				q.push(trie[p][i]);
			}
			else
			{
				trie[p][i]=trie[fail[p]][i];
			}
		}
	}
}
int query()
{
	int p=0,res=0;
	int l=strlen(t+1);
	
	for(int i=1;i<=l;i++)
	{
		p=trie[p][t[i]-'a'];
		for(int j=p;j;j=fail[j])
		//扫描文本串的一个字符，就暴跳一次fail指针。
		//把相对应的最长后缀也给记上。
		//虽然记录的val[j]不一定是一个模式串的尾结点
		//但是我们在确定最大值的时候会通过e[i]来进行判断，只判断尾结点
		//不需要进行去重，因为相对应的模式串在文本串中可能出现很多次。
		{
			++val[j];//直接暴力加就可以
		}
	}
	
	for(int i=1;i<=cnt;i++)
	{
		if(e[i])//先进行判断是不是尾结点
		{
			res=max(res,val[i]);//取最大值
		    num[e[i]]=val[i];//把每个模式串都记录下自己出现了多少次
		}
	}
	
	return res;//最后返回自己的值
}
int main()
{
	while(scanf("%d",&n)&&n)
	{
		init();
		for(int i=1;i<=n;i++)
		{
			scanf("%s\n",s[i]+1);
			insert(strlen(s[i]+1),s[i],i);
		}
		scanf("%s\n",t+1);
		build();
		int mx=query();
		printf("%lld\n",mx);
		for(int i=1;i<=n;i++)
		{
			if(num[i]==mx)//判断出现次数最多的，因为也有可能会出现重复的模式串，他们在文本串中出现的次数很可能是相同的
			{
				printf("%s\n",s[i]+1);//相对应的打印出来就可以
			}
		}
	}
	return 0;
} 

模板三：
#include<bits/stdc++.h>//求解每个模式串在文本串中出现了多少次
#define magic ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
using namespace std;
const int maxn=2e6+10;
int tri[maxn][26];
int vis[maxn],rev[maxn],fail[maxn];
int ans[maxn],res[maxn],indeg[maxn];
int tot;
void Trie(string ms,int id)
{
    int len=ms.length();
    int p=0;
    for(int i=0;i<len;i++)
    {
        if(!tri[p][ms[i]-'a']) tri[p][ms[i]-'a']=++tot;
        p=tri[p][ms[i]-'a'];
    }
    if(!vis[p]) vis[p]=id;
    rev[id]=vis[p];
    return ;
}

void build()
{
    queue<int> q;
    for(int i=0;i<26;i++)
    {
        if(tri[0][i]) q.push(tri[0][i]);
    }

    while(!q.empty())
    {
        int u=q.front();
        q.pop();

        for(int i=0;i<26;i++)
        {
            if(!tri[u][i])
            {
                tri[u][i]=tri[fail[u]][i];
                continue;
            }
            else 
            {
                fail[tri[u][i]]=tri[fail[u]][i];
                indeg[tri[fail[u]][i]]++;
                q.push(tri[u][i]);
            }
        }
    }
    return ;
}
void query(string ts)
{
    int len=ts.length();
    int u=0;

    for(int i=0;i<len;i++)
    {
        int c=ts[i]-'a';
        ans[tri[u][c]]++;
        u=tri[u][c];
    }
}
void topu()
{
    queue<int> q;
    for(int i=1;i<=tot;i++)
    {
        if(!indeg[i]) q.push(i);
    }
    while(!q.empty())
    {
        int ft=q.front();
        q.pop();

        res[vis[ft]]=ans[ft];

        int u=fail[ft];
        ans[u]+=ans[ft];

        if(!(--indeg[u])) q.push(u);
    }
    return ;
}
int main()
{
    magic 
    int n;
    cin>>n;
    string ms;
    for(int i=1;i<=n;i++)
    {
        cin>>ms;
        Trie(ms,i);
    }
    build();
    string ts;
    cin>>ts;
    query(ts);
    topu();
    for(int i=1;i<=n;i++)
    {
        cout<<res[rev[i]]<<"\n";
    }
    return 0;
}
```

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

### 后缀数组

```c++
/*
    Problem: (询问一个字符串中有多少至少出现两次的子串)
    Content: SA's Code and Explanation
    Author : WangYongfeng
*/

/*
    height[i]指的是排名为i的后缀与排名为i-1的后缀的最长公共前缀，即L C P ( i ? 1 , i )
    h[i]指的则是以i为开头的后缀与排名在它前一位的后缀的最长公共前缀，即L C P ( r a n k [ i ] ? 1 , r a n k [ i ] ) LCP(rank[i]-1,rank[i])LCP(rank[i]?1,rank[i])
*/
#include <bits/stdc++.h>

using namespace std;

const int MAXN = 1000005;
// 后缀数组求解时间复杂度是n(log n)^2
char ch[MAXN], All[MAXN];
int SA[MAXN], Rank[MAXN], Height[MAXN], tax[MAXN], tp[MAXN], a[MAXN], n, m;
char str[MAXN]; // 读入的文本串
int H[MAXN];
// rank[i] 第i个后缀的排名; SA[i] 排名为i的后缀位置; Height[i] 排名为i的后缀与排名为(i-1)的后缀的LCP
// tax[i] 基数排序辅助数组; tp[i] rank的辅助数组(基数排序中的第二关键字),与SA意义一样。
// a为原串,n为串的长度
//
void RSort()
{
    // rank第一关键字,tp第二关键字。
    for (int i = 0; i <= m; i++)
        tax[i] = 0;
    for (int i = 1; i <= n; i++)
        tax[Rank[tp[i]]]++;
    for (int i = 1; i <= m; i++)
        tax[i] += tax[i - 1];
    for (int i = n; i >= 1; i--)
        SA[tax[Rank[tp[i]]]--] = tp[i]; // 确保满足第一关键字的同时，再满足第二关键字的要求
} // 数排序,把新的二元组排序。

int cmp(int *f, int x, int y, int w) { return f[x] == f[y] && f[x + w] == f[y + w]; }
// 通过二元组两个下标的比较，确定两个子串是否相同

void Suffix()
{
    // SA
    for (int i = 1; i <= n; i++)
        Rank[i] = a[i], tp[i] = i;
    m = 127, RSort(); // 一开始是以单个字符为单位，所以(m = 127)

    for (int w = 1, p = 1, i; p < n; w += w, m = p)
    { // 把子串长度翻倍,更新rank

        // w 当前一个子串的长度; m 当前离散后的排名种类数
        // 当前的tp(第二关键字)可直接由上一次的SA的得到
        for (p = 0, i = n - w + 1; i <= n; i++)
            tp[++p] = i; // 长度越界,第二关键字为0
        for (i = 1; i <= n; i++)
            if (SA[i] > w)
                tp[++p] = SA[i] - w;

        // 更新SA值,并用tp暂时存下上一轮的rank(用于cmp比较)
        RSort(), swap(Rank, tp), Rank[SA[1]] = p = 1;

        // 用已经完成的SA来更新与它互逆的rank,并离散rank
        for (i = 2; i <= n; i++)
            Rank[SA[i]] = cmp(tp, SA[i], SA[i - 1], w) ? p : ++p;
    }
    // 离散：把相等的字符串的rank设为相同。
    // LCP
    int j, k = 0;
    for (int i = 1; i <= n; Height[Rank[i++]] = k)
        for (k = k ? k - 1 : k, j = SA[Rank[i] - 1]; a[i + k] == a[j + k]; ++k)
            ;
    // 这个知道原理后就比较好理解程序
    // 求H[i];
    for (int i = 1; i <= n; i++)
    {
        H[i] = Height[Rank[i]];
    }
}

void Init()
{
    scanf("%s", str);
    n = strlen(str);
    for (int i = 0; i < n; i++)
        a[i + 1] = str[i];
}

int main()
{
    Init();
    Suffix();
    cout << "SA输出如下" << endl; // SA[i]为第i小的后缀串的起始位置（也就是后缀串的编号）
    for (int i = 1; i <= n; i++)
    {
        printf("%d ", SA[i]);
    }
    printf("\n");
    cout << "Rank输出如下" << endl; // 起始位置为i的后缀串排第几（后缀i的排名）
    for (int i = 1; i <= n; i++)
    {
        printf("%d ", Rank[i]);
    }
    printf("\n");
    cout << "height(排名相邻)输出如下" << endl; // height[1]没有意义。
    for (int i = 1; i <= n; i++)                // 排名为i的后缀与它排名前一个的后缀的最长公共前缀
    {                                           //排名为i的后缀与排名为i-1的后缀的最长公共前缀
        printf("%d ", Height[i]);
    }
    printf("\n");
    cout << "H(顺序临近)输出如下" << endl; // 第i个串的排名与其前一名的LCP
    for (int i = 1; i <= n; i++)
    {
        printf("%d ", H[i]);
    }
    printf("\n");

    return 0;
}
```

### 后缀自动机（SAM）

```c++
#include <cstdio>
#include <cstring>
const int maxn = 2000006;
const int maxc = 27;
int tot = 1, last = 1; // last -> 旧主串
int fa[maxn], len[maxn], size[maxn];
// fa -> fail  fa[x]的right集合一定包含x   fa[x]一定是x的后缀
// len[x] -> x为后缀最长串长度
// size[x] -> x 号节点表示的right集合的大小
// 1 号节点为初始节点
//求出了right集合后在后缀自动机上跑，跑到某个点时，此时的right的大小，亦表示当前匹配到的这个字符串出现次数
int son[maxn][maxc]; // son[p][c] -> 在p所代表的集合后加c字符到，该字符c是哪个节点
//{{{构建SAM
void extend(int c)
{
	int p = last, np = ++tot;//p是旧主串，现在的点是新串，tot表示状态的个数
	last = tot, len[np] = len[p] + 1;//更新旧主串，新状态即是新添加了一个点，长度为加1，是一个连续的转移
	while (p && !son[p][c])//不断的去跳后缀，因为所有对应的后缀都可以添加一个c字符
		son[p][c] = np, p = fa[p]; //跳后缀 它们全都可以加一个c
	//若当前p有c了表示后面所能表示的后缀有节点集合表示了(因为曾经出现过)
	//要么就是所有之前的后缀都需要添加上这个c，要么就是所有之前的后缀都已经有了对应的表示。
	if (!p)
		fa[np] = 1; //表示c从未出现过 它的后缀为空
		//如果出现过的话在上一个的while循环里最终p会跳成1，而不是0.如果是0的话说明之前从来都没有出现过
		//没有出现过就给它赋值成从根点出来的东西。
	else
	{//否则就是这个后缀已经出现过，进行处理。p已经跳到1并且son[p][c]已经有了对应的表示
	//要处理这两个以c为末尾的节点
		int q = son[p][c];//q指这个对应的表示，也就是之前已经存在这个字符串的对应表示。
		if (len[q] == len[p] + 1)//是一个连续的转移
			fa[np] = q; // q是新主串的后缀，连续的转移，可以直接添加
		else//否则就是q会更大，也就是意味着q表示的不仅是x+c这个字符串，还有更长的字符串也是在q的表示中
		{//需要将q状态进行分成两个子状态，需要先进行复制。第一个子状态的长度就是len[p]+1，更新q为这个子状态，然后q的后缀链接指向复制出来的状态

			//即len[q]>len[p]+1
			int nq = ++tot;		  //不是新主串的后缀 因为p是新主串的后缀 而len[q]>len[p]+1且q还没被跳过
			len[nq] = len[p] + 1; // p的endpos多了个n 所以要新节点 表示由p+c得到的后缀 即nq
			fa[nq] = fa[q];		  // nq只是endpos变多 其样子仍是原来那样  其后缀仍是原来的后缀
			fa[np] = fa[q] = nq;  // nq 是 q的后缀 也是新主串的后缀
			memcpy(son[nq], son[q], sizeof(son[q]));
			while (son[p][c] == q)//把所有指向原先q的转移全部都重新指向新建立的clone状态。
				son[p][c] = nq, p = fa[p]; // p的后缀的endpos也多了个n
		}
	}
	size[np] = 1;
}
//}}}
//{{{求right集合大小
int cnt;
int head[maxn], nxt[maxn], to[maxn];
long long ans;
void add(int u, int v)
{
	nxt[++cnt] = head[u], head[u] = cnt, to[cnt] = v;
}
inline long long max(long long x, long long y) { return x > y ? x : y; }
void dfs(int p) //求p所代表的right集合的大小  需先构建出parent树  构建parent树只要让fa[i]向i连边即可  注意根节点为1 应从2开始枚举fa
{
	for (int e = head[p]; e; e = nxt[e])
	{
		dfs(to[e]);
		size[p] += size[to[e]];
	}
	if(size[p]>1) ans=max(ans,1ll*size[p]*len[p]);
}
//}}}
int main()
{
	char s[maxn];
	scanf("%s", s + 1);
	int n = strlen(s + 1);
	for (int i = 1; i <= n; ++i)
		extend(s[i] - 'a' + 1);
	for (int i = 2; i <= tot; ++i)
		add(fa[i], i);
	dfs(1);
	printf("%lld\n",ans);
	return 0;
}
```

## 计算几何

### 点到线段距离

```c++
struct Point {
    // ll x, y;
    db x, y;
};
db GetDistance(Point A, Point B) {
    return sqrt((A.x-B.x) * (A.x-B.x) + (A.y-B.y) * (A.y-B.y));
}
// 求点 A 到线段 BC 的最短距离
db GetNearest(Point A, Point B, Point C) {  // 注意 B、C 两点不能重合
    db a = GetDistance(A, B);
    db b = GetDistance(A, C);
    db c = GetDistance(B, C);
    if (a*a > b*b + c*c)
        return b;
    if (b*b > a*a + c*c)
        return a;
    // db l = (a+b+c) / 2;
    // db s = sqrt(l*(l-a)*(l-b)*(l-c));   // 海伦公式，这里用到了两次 sqrt 函数，会丢失精度
    // return 2*s/c;
    // 利用向量来计算精度不会丢失太多
    db s = 1.0 * fabs(((B.x - A.x) * (B.y - C.y) + (B.y - A.y) * (C.x - B.x)) / c);
    return s;
}
```

## 其他

### 离散化

```c++
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
 
int n;	//原序列长度
int a[1005];//原序列
int len;	//去重后的序列长度
int f[1005];//原序列的副本
 
int main()
{
	scanf("%d",&n);
	for(int i=0;i<n;i++)
	{
		scanf("%d",&a[i]);
		f[i]=a[i];
	}
	sort(f,f+n);//排序
	for(int i=0;i<n;i++)
		printf("%-4d ",a[i]);
	printf("\n");
	len=unique(f,f+n)-f;//去重后的序列长度
	for(int i=0;i<n;i++)	//离散化
		a[i]=lower_bound(f,f+len,a[i])-f;
	for(int i=0;i<n;i++)//用lower_bound 离散化后的结果
		printf("%-4d ",a[i]);
	printf("\n");
	for(int i=0;i<n;i++)
		f[i]=a[i];
	sort(f,f+n);
	len=unique(f,f+n)-f;
	for(int i=0;i<n;i++)	//离散化
		a[i]=upper_bound(f,f+len,a[i])-f;
	for(int i=0;i<n;i++)//用upper_bound 离散化后的结果
		printf("%-4d ",a[i]);
	printf("\n");
	return 0;
}
```

#### 计算乱序到升序最小交换次数

```c++
int getMinSwaps(vector<int> &nums){
    //排序
    vector<int> nums1(nums);
    sort(nums1.begin(),nums1.end());
    unordered_map<int,int> m;
    int len = nums.size();
    for (int i = 0; i < len; i++){
        m[nums1[i]] = i;//建立每个元素与其应放位置的映射关系
    }
 
    int loops = 0;//循环节个数
    vector<bool> flag(len,false);
    //找出循环节的个数
    for (int i = 0; i < len; i++){
        if (!flag[i]){//已经访问过的位置不再访问
            int j = i;
            while (!flag[j]){
                flag[j] = true;
                j = m[nums[j]];//原序列中j位置的元素在有序序列中的位置
            }
            loops++;
        }
    }
    return len - loops;
}
```



