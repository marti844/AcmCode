void push_up(int p)
{
	int ls = 2 * p;
	int rs = 2 * p + 1;
	tree[p] = min(tree[ls],tree[rs]);
}

void build(int p, int l, int r)
{
	
	if(l == r)	{
		tree[p] = a[l];
		return ;
	}
	
	int ls = 2 * p;
	int rs = 2 * p + 1;
	int mid = (l+r)/2;
	
	build(ls,l,mid);
	build(rs,mid+1,r);
	
	push_up(p);
} 
