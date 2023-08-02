# ACM常用模板整理

[toc]

<div STYLE="page-break-after: always;"></div>

## 文件头

```c++
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
```

## 读入输出

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
#include <bits/stdc++.h>
using namespace std;
const int maxn = 500050;
typedef long long ll;
int fa[maxn][40], d[maxn], head[maxn];
int lg[maxn];
int n, m, s;
int cnt;
struct node
{
 int nex, t;
} e[maxn*2];

void add(int x, int y)
{
 e[++cnt].nex = head[x];
 e[cnt].t = y;
 head[x] = cnt;
}

void dfs(int f, int fath) // f表示当前节点，fath表示它的父亲节点
{
 d[f] = d[fath] + 1;
 fa[f][0] = fath;
 for (int i = 1; (1 << i) <= d[f]; i++)
  fa[f][i] = fa[fa[f][i - 1]][i - 1]; //这个转移可以说是算法的核心之一
           //意思是f的2^i祖先等于f的2^(i-1)祖先的2^(i-1)祖先
           // 2^i=2^(i-1)+2^(i-1)
 for (int i = head[f]; i; i = e[i].nex)
  if (e[i].t != fath)
   dfs(e[i].t, f);
}

int lca(int x, int y)
{
 if (d[x] < d[y]) //用数学语言来说就是：不妨设x的深度 >= y的深度
  swap(x, y);
 while (d[x] > d[y])
  x = fa[x][lg[d[x] - d[y]] - 1]; //先跳到同一深度
 if (x == y)         //如果x是y的祖先，那他们的LCA肯定就是x了
  return x;
 for (int k = lg[d[x]] - 1; k >= 0; k--) //不断向上跳（lg就是之前说的常数优化）
  if (fa[x][k] != fa[y][k])    //因为我们要跳到它们LCA的下面一层，所以它们肯定不相等，如果不相等就跳过去。
   x = fa[x][k], y = fa[y][k];
 return fa[x][0]; //返回父节点
}

int main()
{
 cin >> n >> m >> s;
 for (int i = 1; i <= n - 1; i++)
 {
  int x, y;
  cin >> x >> y;
  add(x, y);
  add(y, x);
 }
 for (int i = 1; i <= n; i++)       //预先算出log_2(i)+1的值，用的时候直接调用就可以了
  lg[i] = lg[i - 1] + (1 << lg[i - 1] == i); //看不懂的可以手推一下
 dfs(s, s);
 for (int i = 1; i <= m; i++)
 {
  int x, y;
  cin >> x >> y;
  cout << lca(x, y) << endl;
 }
 system("pause");
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
int qpow(int a, int b) {
    int s = 1;
    for (; b; a = 1ll * a * a % mod, b >>= 1) if (b & 1) s = 1ll * s * a % mod;
    return s;
}
void set_up() {
    frac[0] = inv[0] = 1;
    for (int i = 1; i <= 1000; i++) frac[i] = 1ll * frac[i - 1] * i % mod;
    inv[1000] = qpow(frac[1000], mod - 2);
    for (int i = 999; i; i--)
    inv[i] = 1ll * inv[i + 1] * (i + 1) % mod;
}
inline int C(int n, int m) {
    if (n < m) return 0;
    return 1ll * frac[n] * inv[m] % mod * inv[n - m] % mod;
}
```

### 快速幂

```c++
int qpow(int a, int b) {
    int s = 1;
    for (; b; a = 1ll * a * a % mod, b >>= 1) if (b & 1) s = 1ll * s * a % mod;
    return s;
}
```

### gcd/lmp

```c++
int gcd(int a, int b){
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
long long exgcd(long long a, long long b, long long &x, long long &y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }
    long long  d = exgcd(b, a % b, x, y);
    long long  z = x;
    x = y;
    y = z - a / b * y;
    return d;
}
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



