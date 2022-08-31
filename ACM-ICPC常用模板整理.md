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

## 图论

### 最短路算法

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



