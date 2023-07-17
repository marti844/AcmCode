#include"makedata.h"

void make(int tp){

    // 生成的数据范围

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
    
    /*
    暴力求解的代码
    */     


    outfile<<ans;
}
