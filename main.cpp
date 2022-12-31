#include"makedata.h"

const int N = 20 + 5;
int n;  // 工件数 
int m; // 机器数 
int order[N*N]; // 安排顺序 
int machine[N][N]; // 每个工件的每个工序所使用的机器号 
int cost[N][N]; // 每个工件的每个工序的加工时间 
int st[N][N]; // 每个工件的每个工序最早能开始的时间 
int done[N]; // 每个工件已完成的工序数 
int l[N][N], r[N][N], tot[N]; // 每个机器的空隙，左端点，右端点，和空隙的个数 



void make(int tp){
	times = 20;
    m=num(1,20); n=num(1,20);
    infile<<m<<" "<<n<<endl;
    
    for (int i=1; i<=n*m; i++) order[i]=num(1,20),infile<<order[i]<<" ";
    infile<<endl;
    for (int i=1; i<=n; i++){
        for (int j=1; j<=m; j++) machine[i][j]=num(1,20),infile<<machine[i][j]<<" ";
        infile<<endl;
    }
        
    for (int i=1; i<=n; i++){
        for (int j=1; j<=m; j++) cost[i][j]=num(1,20),infile<<cost[i][j]<<" ";
        infile<<endl;
    }
        
    //以上按题意读入 
    for (int i=1; i<=m; i++) tot[i]=1, l[i][1]=0, r[i][1]=0x3f3f3f3f;
    //初始化空隙，每个机器有一个0到正无穷的空隙 
    for (int k=1; k<=n*m; k++) {
        int job=order[k]; //当前的工作 
        int process=done[job]+1; //通过已完成的工序数能得到当前是第几个工序 
        int number=machine[job][process]; //机器编号 
        for (int i=1; i<=tot[number]; i++) { //枚举空隙 
            if (l[number][i] < st[job][process] && st[job][process]+cost[job][process] <= r[number][i]) {
                //第一种情况考虑当前工序能从最早开工时间开始
                //所以空隙需要包含整个工作过程 
                //如果符合条件，当前空隙会被工作过程再分成两个小空隙，以下即处理 
                tot[number]++; 
                for (int j=tot[number]; j>i+1; j--)
                    l[number][j]=l[number][j-1], r[number][j]=r[number][j-1]; 
                //感觉得把空隙按顺序排好，不按顺序排总感觉会出问题 
                //觉得可以不按顺序排的大佬可以试一试
                //这里我就把后面的空隙往后挪一位咯 
                st[job][process+1]=st[job][process]+cost[job][process];
                //当前工作的下一个工序的最早开工时间也能得出了 
                l[number][i+1]=st[job][process+1];
                //多的一个空隙的左端点即当前工序的结束时间即下一个工序的最早开工时间 
                r[number][i+1]=r[number][i];
                //多的一个空隙的右端点即原空隙的右端点 
                r[number][i]=st[job][process];
                //原空隙的右端点即当前工序的开工时间 
                break;
            }
            else if (l[number][i] >= st[job][process] && l[number][i]+cost[job][process] <= r[number][i]) {
                    //第二种情况考虑当前工序不能从最早开工时间开始
                    //则空隙的大小得大于等于工作的时间 
                    st[job][process+1]=l[number][i]+cost[job][process];
                    //当前工作的下一个工序的时间同样也可以得出了 
                    l[number][i]+=cost[job][process];
                    //左端点往后挪 
                    break;
                 }
        }
        done[job]++; //当前工作完成了一个工序 
    }
    int ans=0;
    for (int i=1; i<=m; i++) ans=max(ans, l[i][tot[i]]);
    //不难想到 
    //每台机器的最后一个空隙是 这台机器上最后一个加工的工序的结束时间到正无穷
    //所以答案即所有机器的最后一个空隙的左端点的最大值 
    printf("%d\n", ans);       
    outfile<<ans;
}
