#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

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
