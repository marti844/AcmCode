//快速幂
long long ksm(int a,int b,int c)
{
    int ans=1;
    a%=c;
    while(b)
    {
        if(b&1)ans=(ans*a)%c;
        a=a*a%c;
        b>>=1;
    }
}
//高精度大数多进制加法
string Plus(string s1,string s2,int n)
{
    int a[10000]={0},b[10000]={0},l1,l2;
    string str;
    l1=s1.length(),l2=s2.length();
    for(int i=0;i<l1;i++){
        if(n!=16)a[i]=s1[l1-i-1]-'0';
        if(n==16){
        if(s1[l1-i-1]=='F')a[i]=16;
        else if(s1[l1-i-1]=='A')a[i]=10;
        else if(s1[l1-i-1]=='B')a[i]=11;
        else if(s1[l1-i-1]=='C')a[i]=12;
        else if(s1[l1-i-1]=='D')a[i]=13;
        else if(s1[l1-i-1]=='E')a[i]=14;
        else a[i]=s1[l1-i-1]-'0';
        }
    }
    for(int i=0;i<l2;i++){
        if(n!=16)b[i]=s2[l2-i-1]-'0';
        if(n==16){
            if(s2[l1-i-1]=='F')b[i]=16;
        else if(s2[l1-i-1]=='A')b[i]=10;
        else if(s2[l1-i-1]=='B')b[i]=11;
        else if(s2[l1-i-1]=='C')b[i]=12;
        else if(s2[l1-i-1]=='D')b[i]=13;
        else if(s2[l1-i-1]=='E')b[i]=14;
        else b[i]=s2[l1-i-1]-'0';
        }
    }
    int lmax=l1>l2?l1:l2;
    for(int i=0;i<lmax;i++){
        a[i]+=b[i];
        a[i+1]+=a[i]/n;
        a[i]=a[i]%n;
    }
    if(!a[lmax])lmax--;
    for(int i=lmax;i>=0;i--){
        if(n!=16)str+=a[i]+'0';
        if(n==16){
            if(a[i]==10)str+='A';
            else if(a[i]==11)str+='B';
            else if(a[i]==12)str+='C';
            else if(a[i]==13)str+='D';
            else if(a[i]==14)str+='E';
            else if(a[i]==15)str+='F';
            else str+=a[i]+'0';
        }
    }
    return str;
}
//快速排序
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
//最大公约数
int gcd(int a, int b){
   return a % b ? gcd(b, a % b) : b;
}
//最小公倍数
int lmp(int a,int b)
{
   return a*b/gcd(a,b);
}
//二分查找
bool search(long long target,long long a[],long long i,long long j)
{
	long long mid=(i+j)/2;
	if(i>j)return false;
	if(a[mid]>target)return search(target,a,i,mid-1);
	else if(a[mid]<target)return search(target,a,mid+1,j);
	else if(a[mid]==target)return true;
}
//Dijkstra(最短路径)
void Dijkstra(int n)
{
    int visit[1001]={0};
    int min,flag;
    visit[1]=1;
    for(int i=1;i<n;i++)
    {
        min=INF;
        flag=1;
        for(int j=1;j<=n;j++)
        {
            if(!visit[j]&&min>dis[j])
            {
                min=dis[j];
                flag=j;
            }
        }
        visit[flag]=1;
        for(int j=1;j<=n;j++)
        {
            if(!visit[j]&&dis[j]>dis[flag]+chess[flag][j])
            dis[j]=dis[flag]+chess[flag][j];
        }
    }
    printf("%d\n",dis[n]);
}
//KMP
{
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

}
//Kruskal(最小生成树)
int Kruskal()
{
    sum=0;
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
//流操作解除绑定
{ios::sync_with_stdio(false);
cin.tie(0);
cout.tie(0);
}
//Prim
{(最小生成树)
const int MAXN=100010;
const int INF=0x3f3f3f3f;
int n, G[MAXV][MAXV]; //n为顶点数，MAXV为最大顶点数
int d[MAXV]; //顶点与集合S的最短距离
bool vis[MAXV] = {false}; //标记数组，vis[i] == true表示访问。初值均为false
int prim() //默认0号为初始点，函数返回最小生成树的边权之和
{
	fill(d, d + MAXV, INF); //fill函数将整个d数组赋为INF
	d[0] = 0; //只有0号顶点到集合S的距离为0，其余全是INF
	int ans = 0; //存放最小生成树的边权之和
	for(int i = 0; i < n; i++) //循环n次
	{
		int u = -1, MIN = INF; //u使d[u]最小，MIN存放该最小的d[u]
		for(int j = 0; j < n; j++) //找到未访问的顶点中d[]最小的
		{
			if(vis[j] == false && d[j] < MIN)
			{
				u = j;
				MIN = d[j];
			}
		}
		//找不到小于INF的d[u]，则剩下的顶点和集合S不连通
		if(u == -1)
			return -1;
		vis[u] = true; //标记u为已访问
		ans += d[u]; //将与集合S距离最小的边加入最小生成树
		for(int v = 0; v < n; v++)
		{
		    //v未访问 && u能到达v && 以u为中介点可以使v离集合S更近
			if(vis[v] == false && G[u][v] != INF && G[u][v] < d[v])
			{
				d[v] = G[u][v]; //将G[u][v]赋值给d[v]
			}
		}
	}
	return ans; //返回最小生成树的边权之和
}
}
//RMQ（st算法）
{
int dpmax[MAXN][20],dpmin[MAXN][20];
int val[MAXN];//目标数组
int len;//数组长度

void rmq_pre(){
	for(int i=1;i<=len;++i)dpmax[i][0]=dpmin[i][0]=val[i];
	for(int j=1;(1<<j)<=len;++j){
		for(int i=1;i+(1<<j)-1<=len;++i){
			dpmin[i][j]=min(dpmin[i][j-1],dpmin[i+(1<<j-1)][j-1]);
			dpmax[i][j]=max(dpmax[i][j-1],dpmax[i+(1<<j-1)][j-1]);
		}
	}
}

int rmq_max(int l,int r){
	int k=log2(r-l+1);
	return max(dpmax[l][k],dpmax[r-(1<<k)+1][k]);
} 

int rmq_min(int l,int r){
	int k=log2(r-l+1);
	return min(dpmin[l][k],dpmin[r-(1<<k)+1][k]);
} 
}
//RMQ（线段树）
{
    void build_tree(int arr[],int tree[],int node,int l,int r){
	if(l==r){
		tree[node]=arr[l];
	} else {
		int mid=(l+r-1)/2;
		int left_node=2*node;
		int right_node=2*node+1;
		
		build_tree(arr,tree,left_node,l,mid);
		build_tree(arr,tree,right_node,mid+1,r);
		
		tree[node]=tree[left_node]+tree[right_node];
	}
}

void tree_update(int arr[],int tree[],int node,int l,int r,int a,int b){
	if(l==r){
		arr[a]+=b;
		tree[node]+=b;
	} else {
		int mid=(l+r-1)/2;
		int left_node=2*node;
		int right_node=2*node+1;
		
		if(a>=l&&a<=mid)tree_update(arr,tree,left_node,l,mid,a,b);
		else tree_update(arr,tree,right_node,mid+1,r,a,b);
		
		tree[node]=tree[left_node]+tree[right_node];
	}
}

int tree_query(int arr[],int tree[],int node,int l,int r,int start,int end){
	
	if(start>r||end<l)return 0;
	else if(l==r)return tree[node];
	else if(l>=start&&r<=end)return tree[node];
	else{
		int mid=(l+r-1)/2;
		int left_node=2*node;
		int right_node=2*node+1;
		
		int sum_left=tree_query(arr,tree,left_node,l,mid,start,end);
		int sum_right=tree_query(arr,tree,right_node,mid+1,r,start,end);
		
		return sum_left+sum_right;
	}
	
}
}
//快速读入
inline int read()
{
	int x=0,y=1;char c=getchar();
	while (c<'0'||c>'9') {if (c=='-') y=-1;c=getchar();}
	while (c>='0'&&c<='9') x=x*10+c-'0',c=getchar();
	return x*y;
}
//欧拉筛法
inline void Euler_Prime(ull x){
    bool is_Prime[MAXN];
    int prime[MAXN+10];
    int cnt = 0;
	memset(is_Prime,true,sizeof is_Prime);
	is_Prime[0] = is_Prime[1] = false;
	for(int i = 2;i<=x;++i){
		if(is_Prime[i])prime[cnt++] = i;
		for(int j = 0;j < cnt; ++j){
			if(i*prime[j]>)break;
			is_Prime[i*prime[j]]=false;
			if(i%prime[j]==0)break;
		}
	}
}

