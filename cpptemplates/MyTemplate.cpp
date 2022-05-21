#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define mod 1000000007
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;

// lenght of longest non-decreasing subsequence
int llnds(vector<int> &v)
{
    vector<int> lis;
    for (auto &x : v)
    {
        auto it = upper_bound(lis.begin(), lis.end(), x);
        if (it == lis.end())
            lis.push_back(x);
        else
            *it = x;
    }
    return lis.size();
    // use lower_bound for longest increasing subsequence
}

// seive of eratostanes
void Sieve(long long n)
{
    vector<bool> isprime(n + 1, true);
    isprime[0] = isprime[1] = false;

    for (int i = 2; i * i <= n; i++)
    {
        if (isprime[i])
        {
            // starting with i*i as i*2,i*3,...i*(i-1) will be already coverd
            // by previous elements
            for (int j = i * i; j <= n; j += i)
                isprime[j] = false;
        }
    }
}

// prime factorisation of a number stored in map
void PrimeFactorisation(ll n)
{
    map<ll, ll> mp;
    while (n % 2 == 0)
    {
        mp[2]++;
        n /= 2;
    }
    for (int i = 3; i * i <= n; i = i + 2)
    {
        while (n % i == 0)
        {
            mp[i]++;
            n /= i;
        }
    }
    if (n > 2)
        mp[n]++;
    for (auto i : mp)
        cout << i.first << " " << i.second << endl;
}

// class for data structure disjoint set union
class Disjoint_set_Union
{
public:
    vector<int> arr;
    vector<int> size;
    int n;
    Disjoint_set_Union(int nn)
    {
        n = nn;
        arr.resize(n);
        size.resize(n);
        for (int i = 0; i < n; i++)
            arr[i] = i;
        for (int i = 0; i < n; i++)
            size[i] = 1;
    }
    int root(int i)
    {
        while (arr[i] != i)
        {
            arr[i] = arr[arr[i]];
            i = arr[i];
        }
        return i;
    }
    void Union(int i, int j)
    {
        int roota = root(i);
        int rootb = root(j);
        if (roota == rootb)
            return;
        if (size[roota] < size[rootb])
        {
            arr[roota] = arr[rootb];
            size[rootb] += size[roota];
        }
        else
        {
            arr[rootb] = arr[roota];
            size[roota] += size[rootb];
        }
    }
};

// single-source shortest path algorithm
int Dijkstra_min_path(int src, int dest, int n, vector<vector<pair<int, int>>> &V)
{
    vector<int> distance(n);
    vector<int> before(n, -1);
    vector<int> visited(n, 0);
    set<pair<int, int>> s; // set for maintaing {distance to vertext,vertext} to get the closest vertext
    s.insert({0, src});    // insert src with distance 0
    while (!s.empty())
    {
        // pick shortest vertext
        pair<int, int> g = (*s.begin());
        s.erase(s.begin()); // remove from set
        ll u = g.second, curr_dist = g.first;
        if (visited[u])
            continue; // if visited current node then continue
        visited[u] = 1;
        distance[u] = curr_dist;

        for (int i = 0; i < V[u].size(); i++)
        {
            // insert all non-visited vertices in set,
            ll v = V[u][i].first;
            ll w = V[u][i].second;

            if (visited[v] == 0)
            {
                before[v] = u;
                s.insert({curr_dist + w, v});
            }
        }
    }
    return distance[dest]; // return shortest_path to destination
}

// segment tree implementation for sum of range
class Segment_Tree
{
private:
    vector<ll> tree;
    ll tree_size;

public:
    Segment_Tree(int n) // constructor
    {
        tree_size = n;
        tree.resize(4 * n, 0);
    }

    // build tree from array
    void build(vector<ll> &arr, ll v, ll tl, ll tr)
    {
        if (tl == tr)
            tree[v] = arr[tl];
        else
        {
            ll tm = (tl + tr) / 2;
            build(arr, 2 * v + 1, tl, tm);
            build(arr, 2 * v + 2, tm + 1, tr);
            tree[v] = tree[2 * v + 1] + tree[2 * v + 2];
        }
    }

    // get sum of l to r range
    ll sum(ll v, ll tl, ll tr, ll l, ll r)
    {
        if (l > r)
            return 0ll;
        if (l == tl && r == tr)
        {
            return tree[v];
        }
        ll tm = (tl + tr) / 2;
        return sum(v * 2 + 1, tl, tm, l, min(r, tm)) + sum(v * 2 + 2, tm + 1, tr, max(l, tm + 1), r);
    }

    // update a position with new value
    void update(ll v, ll tl, ll tr, ll pos, ll new_val)
    {
        if (tl == tr)
        {
            tree[v] = new_val;
        }
        else
        {
            ll tm = (tl + tr) / 2;
            if (pos <= tm)
                update(v * 2 + 1, tl, tm, pos, new_val);
            else
                update(v * 2 + 2, tm + 1, tr, pos, new_val);
            tree[v] = tree[v * 2 + 1] + tree[v * 2 + 2];
        }
    }
};

class Lazy_Segment_Tree
{
private:
    vector<ll> tree;
    ll tree_size;
    vector<ll> lazy;

public:
    Lazy_Segment_Tree(ll n) // constructor
    {
        tree_size = n;
        tree.resize(4 * n, 0);
        lazy.resize(4 * n, 0);
    }

    // update
    // This is for addition update
    void push(ll v, ll tl, ll tr)
    {
        if (lazy[v] != 0)
        {
            // update the current node
            tree[v] += lazy[v] * (tr - tl + 1);
            if (!(2 * v + 1 < 4 * tree_size and (2 * v + 2) < 4 * tree_size))
            {
                lazy[v] = 0;
                return;
            }

            // pass value to be added to children
            lazy[2 * v + 1] += lazy[v];
            lazy[2 * v + 2] += lazy[v];
            lazy[v] = 0;
        }
    }

    // build tree from array
    void build(vector<ll> &arr, ll v, ll tl, ll tr)
    {
        if (tl == tr)
            tree[v] = arr[tl];
        else
        {
            ll tm = (tl + tr) / 2;
            build(arr, 2 * v + 1, tl, tm);
            build(arr, 2 * v + 2, tm + 1, tr);
            tree[v] = tree[2 * v + 1] + tree[2 * v + 2];
        }
    }

    void print_tree(ll v, ll tl, ll tr)
    {
        push(v, tl, tr);
        if (tl == tr)
        {
            cout << "index " << tl << " : " << tree[v];
            cout << endl;
        }
        else
        {
            ll tm = (tl + tr) / 2;
            print_tree(2 * v + 1, tl, tm);
            print_tree(2 * v + 2, tm + 1, tr);
        }
    }
    void get(ll v, ll tl, ll tr, ll x, vector<ll> &ans)
    {
        push(v, tl, tr);
        if (tl == tr)
        {
            ans.push_back(tree[v]);
            return;
        }
        ll tm = (tl + tr) / 2;
        get(2 * v + 1, tl, tm, x, ans);
        get(2 * v + 2, tm + 1, tr, x, ans);
    }

    // update a position with new value
    void update(ll v, ll tl, ll tr, ll l, ll r, ll new_val)
    {
        if (l > r)
            return;
        if (l == tl && r == tr)
        {
            lazy[v] += new_val;
        }
        else
        {
            push(v, tl, tr);
            ll tm = (tl + tr) / 2;
            update(v * 2 + 1, tl, tm, l, min(tm, r), new_val);
            update(v * 2 + 2, tm + 1, tr, max(tm + 1, l), r, new_val);
        }
    }
};

// implementation for upper_bound
ll Upper_bound(ll n, ll val, Segment_Tree &st)
{
    ll low = 0ll, high = n - 1;
    ll ans = 0;
    while (low <= high)
    {
        ll mid = (low + high) / 2;
        ll curr = st.sum(0ll, 0ll, n - 1, 0ll, mid);
        if (val >= curr)
        {
            ans = mid;
            low = mid + 1;
        }
        else
            high = mid - 1;
    }
    return ans + 1;
}

class Fenwick
{
private:
    vector<ll> Binary_index_tree;
    ll N;

public:
    Fenwick(ll n)
    {
        N = n;
        Binary_index_tree.assign(N, 0);
    }
    Fenwick(vector<ll> arr) : Fenwick(arr.size())
    {
        for (ll i = 0; i < arr.size(); i++)
        {
            add(i, arr[i]);
        }
    }
    void add(ll index, ll change)
    {
        while (index < N)
        {
            Binary_index_tree[index] += change;
            index = index | (index + 1); // to flip last unset bit, that is last 0
        }
    }
    ll sum(ll r)
    {
        ll val = 0;
        while (r >= 0)
        {
            val += Binary_index_tree[r];
            r = (r & (r + 1)) - 1; // replace all trailing 1 bits to 0,i.e. that 1's after first 0
        }
        return val;
    }
    ll sum(ll l, ll r)
    {
        return sum(r) - sum(l - 1);
    }
};

// Sparse tables
// for immutable arrays
class SparseTable
{
    ll N;
    int k; // k value won't be that big
    vector<vector<ll>> STable;

public:
    SparseTable(vector<ll> arr, ll n)
    {
        N = n;
        k = (int)log2(N);
        STable.resize(N, vector<ll>(k + 1));
        for (ll i = 0; i < N; i++)
        {
            STable[i][0] = arr[i];
        }
        for (ll j = 1; j <= k; j++)
            for (ll i = 0; i + (1ll << j) <= N; i++)
                STable[i][j] = min(STable[i][j - 1], STable[i + (1ll << (j - 1))][j - 1]);
    }
    ll idempotent_query(ll l, ll r)
    {
        ll ans = 1e9;
        ll j = log2(r - l + 1);
        /* O(1) solution for range queries. This works only for
        idempotent functions, that is functions that can be applied multiple times or upon overlapping intervals,
        and final answer will be same. It won't work for Sum, Product,etc (i.e.: Non-idempotent functions) */
        ans = min(STable[l][j], STable[r - (1 << j) + 1][j]);
        return ans;
    }
    ll nonidempotent_query(ll l, ll r)
    {
        // works for non-idempotent queries like sum
        // but time is O(log(N))
        ll sum = 0;
        for (int j = k; j >= 0; j--)
        {
            if ((1ll << j) <= (r - l + 1))
            {
                sum += STable[l][j];
                l += (1 << j);
            }
        }
        return sum;
    }
};

// Rabin-Karp algorithm for pattern matching
// Be consious while using this, it is unreliable sometimes
vector<int> Rabin_Karp(string const &s, string const &t)
{
    // constants for computing polynomial rolling hash function
    const int p = 31;
    const int m = 1e9 + 9;
    // here s is pattern, t is text
    int S = s.size(), T = t.size();
    vector<ll> p_pow(max(S, T));
    p_pow[0] = 1;
    for (int i = 1; i < (int)p_pow.size(); i++)
        p_pow[i] = (p_pow[i - 1] * p) % m;

    vector<ll> HashText(T + 1, 0); // vector to store hash upto index i
    for (int i = 0; i < T; i++)
        HashText[i + 1] = (HashText[i] + (t[i] - 'a' + 1) * p_pow[i]) % m;
    ll pattern_hash = 0;
    for (int i = 0; i < S; i++)
        pattern_hash = (pattern_hash + (s[i] - 'a' + 1) * p_pow[i]) % m;

    vector<int> occurences;
    for (int i = 0; i + S - 1 < T; i++)
    {
        ll curr_hash = (HashText[i + S] + m - HashText[i]) % m;
        if (curr_hash == pattern_hash * p_pow[i] % m)
            occurences.push_back(i);
    }
    return occurences;
}

/*
    => Precomputation helps in some cases to obtain better performances.

*/

// Kadane's Algorithm for maximum(or minimum) sum subarray
ll kadanes(vector<ll> arr, int n)
{
    ll ans = arr[0];
    ll ans_l = 0, ans_r = 0;
    ll sum = 0, minus_pos = -1;
    for (int r = 0; r < n; r++)
    {
        sum += arr[r];
        if (sum > ans)
        {
            ans = sum;
            ans_l = minus_pos + 1;
            ans_r = r;
        }
        if (sum < 0)
        {
            sum = 0;
            minus_pos = r;
        }
    }
    return ans;
}
ll Max_sum_subarray_algo2(vector<ll> arr, int n)
{
    ll ans = arr[0];
    ll ans_l = 0, ans_r = 0;
    ll sum = 0, min_sum = 0, min_r = 0;
    for (int r = 0; r < n; r++)
    {
        sum += arr[r];
        if (sum - min_sum > ans)
        {
            ans = sum - min_sum;
            ans_l = min_r + 1;
            ans_r = r;
        }
        if (sum < min_sum)
        {
            min_sum = sum;
            min_r = r;
        }
    }
    return ans;
}

int find_pivot(vector<ll> arr)
{
    int n=arr.size();
    int low=0,high=n-1;
    int mid;
    while (low<=high)
    {
       if(low==high) return low;
       mid=(high+low)/2;
       if(mid<high and arr[mid]>arr[mid+1]) return mid;
       if(mid>low and arr[mid]<arr[mid-1]) return mid-1;
       if(arr[low]>=arr[mid]) high=mid-1;
       else low=mid+1;
    }
    return -1;
}

ll binary_search_pivot(vector<ll> arr,ll key)
{
   int n=arr.size();
   int low=0,high=n-1;
   while(low<=high)
   {
       int mid=(low+high)/2;
       if(arr[mid]==key) return mid;
       
       // if lower half is sorted
       if(arr[low]<=arr[mid])
       {
           // if key lies in lower half
           if(arr[low]<=key and arr[mid]>=key) high=mid-1;
           else low=mid+1;
       }
       // if lower half is not sorted, upper half must be
       else 
       {
           // if key lies in upper half
           if(arr[mid]<=key and key<=arr[high]) low=mid+1;
           else high=mid-1;
       }
   }   
       return -1;
}