#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define mod 1000000007
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef tuple<int, int, int> tiii;

// To include ordered set
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
#define ordered_set tree<int, null_type, less, rb_tree_tag, tree_order_statistics_node_update>


/*******************************************************************************/
/********************************** STANDARD ALGORITMS *************************/
/*******************************************************************************/
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

ll gcd(ll a, ll b)
{
    if (a == 0)
        return b;
    return gcd(b % a, a);
}

//  Extended Euclidean algorithm also finds the co-efficients
//  x,y such that xa+yb=gcd(a,b);
pll Egcd(ll a, ll b, ll &gcd)
{
    if (a == 0)
    {
        gcd = b;
        return {0, 1};
    }
    pll t = Egcd(b % a, a, gcd);
    return {t.second - t.first * (b / a), t.first};
}
// let a*x+b*y=gcd , then (b%a)*x1+a*y1=gcd => (b-floor(b/a)*a)*x1+a*y1=gcd
// by equating, x=y1-floor(b/a)*x1 , y=x1

// n inverse mod m = x such that xn+ym=1 (gcd will be 1 as m is prime)
ll mod_inverse_euclidean(ll n, ll m)
{
    ll gcd = 0; // to get the gcd
    return (Egcd(n % m, m, gcd).first + m) % m;
}

ll mod_inverse_iterative(ll n, ll m)
{
    ll m0 = m;
    ll x = 0, y = 1; // co-effieceints in base-case of Egcd

    if (m == 1)
        return 0; // trivial

    while (n > 1)
    {
        ll q = n / m;
        ll t = m;

        m = n % m, n = t;
        t = x;
        x = y - q * x;
        y = t;
    }
    return (y + m0) % m0;
}

// works only when m is prime
// Fermat's little theorem , a^(m-1)=1 (mod m)
// so we need to find a^(m-2)
ll mod_inverse_fermats(ll n, ll m)
{
    ll r=1;
    ll p=m-2;
    while(p)
    {
        if(p&1) r=(r*n)%m;
        n=(n*n)%m;
        p>>=1;
    }
    return r;
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

// implementation of upper bound
ll upper_bound(vector<ll> arr, ll x)
{
    ll n = arr.size();
    ll low = 0, high = n - 1;
    if (x >= arr[high])
        return -1;
    ll mid;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (arr[mid] <= x)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return arr[low];
}

// implementation of lower bound
ll lower_bound(vector<ll> arr, ll x)
{
    ll n = arr.size();
    ll low = 0, high = n - 1;
    ll mid;
    if (x > arr[high])
        return -1;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (arr[mid] >= x)
            high = mid - 1;
        else
            low = mid + 1;
    }
    return arr[high + 1];
}

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

// KMP algorithm for pattern matching
vector<int> computeLPSArray(string pat)
{
    vector<int> lps;
    int m = pat.size();
    lps[0] = 0;  // only proper prefixes are allowed
    int len = 0; // length of lps
    int i = 1;   // current index;
    while (i < m)
    {
        // example "abababca" and i==5, len==3. The longest prefix suffix is "aba", when pat[i]==pat[len],
        // we get new prefix "abab" and new suffix "abab", so increase length of  current lps by 1 and go to next iteration.
        if (pat[i] == pat[len])
        {
            len++;
            lps[i] = len;
            i++;
        }
        else // (pat[i] != pat[len])
        {
            if (len != 0)
            {
                len = lps[len - 1];
                // This is tricky. Consider the example "ababe......ababc", i is index of 'c', len==4. The longest prefix suffix is "abab",
                // when pat[i]!=pat[len], we get new prefix "ababe" and suffix "ababc", which are not equal.
                // This means we can't increment length of lps based on current lps "abab" with len==4. We may want to increment it based on
                // the longest prefix suffix with length < len==4, which by definition is lps of "abab". So we set len to lps[len-1],
                // which is 2, now the lps is "ab". Then check pat[i]==pat[len] again due to the while loop, which is also the reason
                // why we do not increment i here. The iteration of i terminate until len==0 (didn't find lps ends with pat[i]) or found
                // a lps ends with pat[i].
            }
            else // if (len == 0)
            {    // there isn't any lps ends with pat[i], so set lps[i] = 0 and go to next iteration.
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}

void KMPSearch(string pat, string text)
{
    int m = pat.length();
    int n = text.length();
    vector<int> lps = computeLPSArray(pat);
    vector<int> indices;
    int i = 0, j = 0;
    while (i < n)
    {
        if (pat[j] == text[i])
        {
            j++;
            i++; // if match, increse pointers
        }
        if (j == m)
        {
            // insert starting index in text
            indices.push_back(i - j);
            j = lps[j - 1];
        }
        else if (i < n and pat[j] != text[i]) // mismatch after jmatches
        {
            // but a suffix upto j-1 length is already matched,
            // if there is a prefix equal to suffix, then no need to match it again
            // Do not match pat[0....lps[j-1]] , they are already matched
            if (j != 0)
                j = lps[j - 1];
            // if no character is matched before, then increase i
            else
                i++;
        }
    }
}

// a O(n) appraoch to find max 3 numbers
// and min2 numbers , without using sorting
int maximumProduct(vector<int> &nums)
{
    int min1 = INT_MAX, min2 = INT_MAX;
    int max1, max2, max3;
    max1 = max2 = max3 = INT_MIN;
    for (int i : nums)
    {
        if (i <= min1)
        {
            min2 = min1;
            min1 = i;
        }
        else if (i <= min2)
            min2 = i;
        if (i >= max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = i;
        }
        else if (i >= max2)
        {
            max3 = max2;
            max2 = i;
        }
        else if (i > max3)
            max3 = i;
    }
    return max(max1 * min1 * min2, max1 * max2 * max3);
}

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
    int n = arr.size();
    int low = 0, high = n - 1;
    int mid;
    while (low < high)
    {
        mid = (high + low) / 2;
        if (mid < high and arr[mid] > arr[mid + 1])
            return mid;
        if (mid > low and arr[mid] < arr[mid - 1])
            return mid - 1;
        if (arr[low] >= arr[mid])
            high = mid - 1;
        else
            low = mid + 1;
    }
    return -1;
}

ll binary_search_pivot(vector<ll> arr, ll key)
{
    int n = arr.size();
    int low = 0, high = n - 1;
    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (arr[mid] == key)
            return mid;

        // if lower half is sorted
        if (arr[low] <= arr[mid])
        {
            // if key lies in lower half
            if (arr[low] <= key and arr[mid] >= key)
                high = mid - 1;
            else
                low = mid + 1;
        }
        // if lower half is not sorted, upper half must be
        else
        {
            // if key lies in upper half
            if (arr[mid] <= key and key <= arr[high])
                low = mid + 1;
            else
                high = mid - 1;
        }
    }
    return -1;
}

void next_perm(vector<ll> &arr)

{
    int i;
    // LeetCode - Next Permutation solution

    for (i = arr.size() - 1; i > 0; i--)
    {
        if (arr[i - 1] < arr[i])
        {
            int mn = i;
            for (int j = i; j < arr.size(); j++)
                if (arr[j] > arr[i - 1] and arr[j] < arr[mn])
                    mn = j;
            swap(arr[mn], arr[i - 1]);
            break;
        }
    }
    sort(arr.begin() + i, arr.end());
}

// returns the element that occurs more than n/2 times if there is any
void MooresVotingAlog(vector<ll> arr)
{
    int n = arr.size();
    int min_index = 0, count = 1;
    for (int i = 1; i < n; i++)
    {
        if (arr[i] == arr[min_index])
            count++;
        else
            count--;

        // This happens when we found that number of elements equal to current element
        // is same as number of elements that are not equal to current
        // so we change our decision
        if (count == 0)
        {
            min_index = i;
            count = 1;
        }
    }
    // return arr[min_index];
}

vector<long long> nextLargerElement(vector<long long> arr, int n)
{
    stack<long long> st;
    vector<long long> ng(n);
    for (int i = n - 1; i >= 0; i--)
    {
        while (!st.empty() and st.top() < arr[i])
            st.pop();
        if (st.empty())
            ng[i] = -1;
        else
            ng[i] = st.top();
        st.push(arr[i]);
    }
    return ng;
}

void counting_sort(int arr[], int n, int range)
{
    int count[range] = {0};
    for (int i = 0; i < n; i++)
        count[arr[i]]++;
    for (int i = 1; i < range; i++)
        count[i] += count[i - 1];
    int temp[n];
    for (int i = 0; i < n; i++)
    {
        temp[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }
    // temp is now sorted
}

void heapify(int arr[], int n, int i)
{
    // heapify elements in arr starting from  i to n
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    if (left < n and arr[largest] < arr[left])
        largest = left;
    if (right < n and arr[largest] < arr[right])
        largest = right;
    if (largest != i)
    {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heap_sort(int arr[], int n)
{
    // build heap
    for (int i = n / 2 - 1; i >= 0; i--) // this covers all non-leaf nodes
    {
        heapify(arr, n, i);
    }
    for (int i = n - 1; i > 0; i--)
    {
        // move highest elment to ith position ans heapify upto i-1
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

/*****************************************************************************/
/**************************** SPECIAL DATA STRUCTURES ************************/
/*****************************************************************************/

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
    // v is the node number for the range [tl,tr]
    void build(vector<ll> &arr, ll v, ll tl, ll tr)
    {
        if (tl == tr)
            tree[v] = arr[tl];
        else
        {
            ll tm = (tl + tr) / 2;
            build(arr, 2 * v + 1, tl, tm);
            build(arr, 2 * v + 2, tm + 1, tr);
            tree[v] = tree[2 * v + 1] + tree[2 * v + 2]; // for sum
        }
    }

    // get sum of l to r range
    // tl-tr is range of node v and not the range of sum we check
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

    void push(ll v, ll tl, ll tr)
    {
        if (lazy[v] != 0)
        {
            // pass value to be added to children
            lazy[2 * v + 1] += lazy[v];
            tree[2 * v + 1] += lazy[v] * ((tl + tr) / 2 - tl + 1);
            lazy[2 * v + 2] += lazy[v];
            tree[2 * v + 2] += lazy[v] * (tr - (tl + tr) / 2);
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

    // get sum of l to r range
    // tl-tr is range of node v and not the range of sum we check
    ll sum(ll v, ll tl, ll tr, ll l, ll r)
    {
        if (l > r)
            return 0ll;
        if (l == tl && r == tr)
        {
            return tree[v];
        }
        push(v, tl, tr);
        ll tm = (tl + tr) / 2;
        return sum(v * 2 + 1, tl, tm, l, min(r, tm)) + sum(v * 2 + 2, tm + 1, tr, max(l, tm + 1), r);
    }

    // update a range with new value
    void update(ll v, ll tl, ll tr, ll l, ll r, ll new_val)
    {
        if (l > r)
            return;
        if (l == tl && r == tr)
        {
            tree[v] += new_val * (r - l + 1);
            lazy[v] += new_val;
        }
        else
        {
            push(v, tl, tr);
            ll tm = (tl + tr) / 2;
            update(v * 2 + 1, tl, tm, l, min(tm, r), new_val);
            update(v * 2 + 2, tm + 1, tr, max(tm + 1, l), r, new_val);
            tree[v] = tree[2 * v + 1] + tree[2 * v + 2];
        }
    }
};

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
            STable[i][0] = arr[i];  // range for [i,i+2^j-1]
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

class Trie
{
    struct Trie_Node
    {
        int next[26];
        bool isEndOfWord = false;
        Trie_Node()
        {
            fill(next, next + 26, -1);
        }
    };

    vector<Trie_Node> trie;

public:
    void insert(string new_string)
    {
        if (trie.size() == 0)
            trie.emplace_back();
        int last = 0;
        for (auto ch : new_string)
        {
            int c = ch - 'a';
            if (trie[last].next[c] == -1)
            {
                trie[last].next[c] = trie.size();
                trie.emplace_back();
            }
            last = trie[last].next[c];
        }
        trie[last].isEndOfWord = true;
    }
    bool search(string target)
    {
        if (trie.size() == 0)
            return false;
        int last = 0;
        for (auto ch : target)
        {
            int c = ch - 'a';
            if (trie[last].next[c] == -1)
                return false;
            last = trie[last].next[c];
        }
        return trie[last].isEndOfWord;
    }
};

int partition(vector<int> &arr, int low, int high)
{
    // selecting pivot as the last element
    int i = low - 1, pivot = arr[high];
    for (int j = low; j < high; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1; // it contains the pivot element
}
void quick_sort(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high); // Find the partition
        quick_sort(arr, low, pi - 1);       // sort left
        quick_sort(arr, pi + 1, high);      // sort right
    }
}

/*
    => Precomputation helps in some cases to obtain better performances.
    => Consecutive numbers are co-primes
    => x&1 is 0 if x is even and x&1 is 1 is x is odd.
    => More Generally X is divisble by 2^k exactly when x&(2^k -1)=0
    => ~x(not of x) = -x -1 ( from 2's complement procedure)
    => use left shifts << and right shifts >> when multiplying or dividing with 2^k
    => x= x|(1<<k)  to set kth bit of x to 1
    => x= x&(~(1<<k)) to set kth bit of x to 0
    => x=x^(1<<k) inverts the kth bit of
    => x=x&(x-1) changes last set bit of x to 0
    => x=x&~x sets all one bits to 0 , expect the last one bit
    => if x&(x-1) =0, then x is a power of two
    => b=0; do {}while(b=(b-x)&x) willl process all subsets of x
    => Handle subtractions while using modulo operations as (a-b+mod)%mod
*/

/*****************************************************************************/
/************************************Graphs***********************************/
/*****************************************************************************/
void bfs(vector<int> adj[], vector<int> &dist, vector<bool> &visited, vector<int> &parent, int source)
{
    queue<int> q;
    q.push(source);
    visited[source] = true;
    parent[source] = -1;
    // use while parent[i]!=-1 , to extract path
    while (!q.empty())
    {
        int v = q.front();
        q.pop();
        // Process node here
        for (int u : adj[v])
            if (!visited[u])
            {
                visited[u] = true;
                q.push(u);
                parent[u] = v;
                dist[u] = dist[v] + 1;
            }
    }
}

// dfs finds the lexographically first path
//  if graph is tree, dfs finds shortest path
void dfs(vector<int> adj[], vector<bool> &visited, int curr)
{
    visited[curr] = true;
    // Process Node here
    for (int u : adj[curr])
        if (!visited[u])
            dfs(adj, visited, u);
}

// detect cycle in undirected graph
// for all verticies if(!visited[i]){ if(check(i)) return true;}
bool undirected_isCyclic(int curr, vector<int> adj[], vector<bool> &visited, int parent)
{
    visited[curr] = true;
    for (auto j : adj[curr])
    {
        if (!visited[curr])
        {
            if (undirected_isCyclic(j, adj, visited, curr))
                return true;
        }
        else if (curr != parent)
            return true;
    }
    return false;
}

// detect cycle in a directed graph
// for all verticies if(!visited[i]){ if(check(i)) return true;}
bool directed_isCyclic(int curr, vector<int> adj[], vector<bool> &visited, vector<int> &rec_stack)
{
    visited[curr] = true;
    rec_stack[curr] = true;
    for (auto j : adj[curr])
    {
        if (!visited[j])
        {
            if (directed_isCyclic(j, adj, visited, rec_stack))
                return true;
        }
        else if (rec_stack[j] == true)
            return true;
    }
    rec_stack[curr] = false;
    return false;
}

// Bellman Ford Algo: Single source shortest path with negative edges(without negative weight cycles)
// required list of edges
// Alternate algorithm: SPFA (Google it :)
void bellman_ford(int V, int source, vector<int> &distances, vector<tuple<int, int, int>> edges, vector<int> &parent)
{
    vector<int> path;
    distances.resize(V, INT_MAX);
    parent.resize(V, -1);
    distances[source] = 0;
    int m = edges.size();
    bool updated = false;
    // shortest path can't have more than n-1 edges in a graph of n vertices
    // therfore running n-1 times is enough as in each iteration, atleast one edge
    // in the shortest path to any vertex will be updated
    // If updating continued in nth iteration, then negative cycle is present!!!
    for (int i = 0; i < V - 1; i++)
    {
        updated = false;
        for (int j = 0; j < m; j++)
        {
            int a, b, c;
            tie(a, b, c) = edges[j];
            if (distances[a] < INT_MAX)
            {
                updated = true;
                parent[b] = a;
                distances[b] = min(distances[b], distances[a] + c);
            }
        }
        if (!updated)
            break;
    }
    // to find path to a target t;
    // if(distances[t]==INT_MAX) cout<<"No path exists\n";
    // for(int curr=t;curr!=-1;curr=parent[curr]) path.push_back(curr);
    // reverse(path.begin(),path.end());
}

// single-source shortest path algorithm
int Dijkstra_min_path(int src, int dest, int n, vector<pair<int, int>> V[])
{
    vector<int> distance(n);
    vector<int> before(n, -1);
    vector<int> visited(n, 0);
    // priority queue for maintaing {distance to vertext,vertext} to get the closest vertex
    // This is faster than set
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, src}); // insert src with distance 0
    visited[0] = 1;
    while (!pq.empty())
    {
        // pick shortest vertext
        pair<int, int> g = pq.top();
        pq.pop();
        ll u = g.second, curr_dist = g.first;
        if (visited[u])
            continue; // if visited current node then continue
        distance[u] = curr_dist;

        for (int i = 0; i < V[u].size(); i++)
        {
            // insert all non-visited vertices in set,
            ll v = V[u][i].first;
            ll w = V[u][i].second;

            if (visited[v] == 0)
            {
                before[v] = u;
                visited[v] = 1;
                pq.push({curr_dist + w, v});
            }
        }
    }
    return distance[dest]; // return shortest_path to destination
}

// all-pair shortest path algorithm
void Floyd_Warshall_algo(int V, vector<vector<int>> &adj, vector<vector<int>> &dist)
{
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            if (i == j)
                dist[i][j] = 0;
            else if (adj[i][j])
                dist[i][j] = adj[i][j];
            else
                dist[i][j] = INT_MAX;
        }
    }
    // now
    for (int k = 0; k < V; k++)
    {
        for (int i = 0; i < V; i++)
        {
            for (int j = 0; j < V; j++)
            {
                if (dist[k][j] < INT_MAX and dist[i][k] < INT_MAX)
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
}

// From Cayley's formula, the number of trees possible
// on n-labelled vertices in n^(n-2)

void MST_prims(int V, vector<pair<int, int>> adj[])
{
    vector<pair<int, int>> mst[V];

    // tuple with first value, weight,second - starting vertex of edge, third-ending vertex of edge
    priority_queue<tiii, vector<tiii>, greater<tiii>> pq;

    // push all edge into priority queue
    for (auto i : adj[0])
        pq.push(make_tuple(i.second, 0, i.first));
    while (!pq.empty())
    {
        int w, u, v;
        tie(w, u, v) = pq.top();
        pq.pop();
        // If current vertex is already in mst, then continue
        if (mst[v].size() != 0)
            continue;
        mst[u].push_back({v, w});
        mst[v].push_back({u, w});
        for (auto i : adj[v])
        {
            // if adjacent vertex is not in mst, then push the edge
            if (mst[i.first].size() == 0)
            {
                pq.push(make_tuple(i.second, v, i.first));
            }
        }
    }
}

// Kruskal's algorithm : using disjoint set union
void kruskals(int V, vector<tiii> adj)
{
    // edge list representation
    // edge = {weight,startvertex, endvertex}
    sort(adj.begin(), adj.end());
    Disjoint_set_Union dsu(V);
    vector<tiii> result;
    for (auto e : adj)
    {
        int u, v, w;
        tie(w, u, v) = e;
        if (dsu.root(u) != dsu.root(v))
        {
            result.push_back(e);
            dsu.Union(u, v);
        }
    }
}

// only for directed graphs
void topological_sort(int curr, vector<int> adj[], vector<int> &result, vector<int> &state, bool &cyclic_flag)
{
    if (cyclic_flag)
        return;
    if (state[curr] == 1)
    {
        cyclic_flag = true; // Cycle is detected
        return;
    }

    state[curr] = 1;
    for (auto i : adj[curr])
    {
        if (state[i] == 1)
        {
            cyclic_flag = true;
            return;
        }
        if (state[i] == 0)
            topological_sort(i, adj, result, state, cyclic_flag);
        if (cyclic_flag)
            return;
    }
    state[curr] = 2;
    result.push_back(curr + 1);
}

// Floyd's Algorithm to detect a cycle in a successor graph (or a Linked-list )
// It finds the length of the cycle and a vertex in it
void Floyds(vector<int> succ, int n)
{
    // start at a random point
    int src = 0;
    int a, b;
    a = succ[src];
    b = succ[succ[src]];
    while (a != b)
    {
        a = succ[a];
        b = succ[succ[b]];
    }

    // find the first vertex in cycle
    a = src;
    while (a != b)
    {
        a = succ[a];
        b = succ[b];
    }
    int first_vertex = a;

    // find the length of the cycle
    b = succ[a];
    int length = 1;
    while (a != b)
    {
        b = succ[b];
        length++;
    }
}

// Strong connectivity: A path exists from every node to every other node
// in a "directed" graph. KOSARAJU'S ALGORITHM
void KosarajuSAlgo(vector<int> adj[], int V)
{
    // TODO
}

// to use custom comparator
// map<int,int> mp;
// auto comp=[&mp](int n1,int n2){return mp[n1]>mp[n2];};
// priority_queue<int,vector<int>,decltype(comp)> heap(comp);

/***************************************************************************************/
/**************************Dynamic Programming *****************************************/
/***************************************************************************************/

void knapsack(int W, int wt[], int val[], int n)
{
    // dp[i][w] = maximum value with a knapsack of weight w and considering 1st i items
    // dp[i][w]=max(dp[i-1][w],dp[i-1][w-wi]+val[i])
    // It is maximum when we pick current element, and when we won't pick current element

    int dp[2][W + 1]; // we only need current and previous row
    for (int i = 0; i <= n; i++)
    {
        for (int w = 0; w <= w; w++)
        {
            if (w == 0 || i == 0)
                dp[i % 2][w] = 0;
            else
            {
                if (wt[i] <= w)
                {
                    dp[i % 2][w] = max(dp[(i - 1) % 2][w], dp[(i - 1) % 2][w - wt[i]] + val[i]);
                }
                else
                    dp[i % 2][w] = dp[(i - 1) % 2][w];
            }
        }
    }
    // ans = dp[n%2][W]
}


// Some useful lines
// Common memset settings
//memset(memo, -1, sizeof memo); // initialize DP memoization table with -1
//memset(arr, 0, sizeof arr); // to clear an array of integers
// ans = a ? b : c; // to simplify: if (a) ans = b; else ans = c;
// ans += val; // to simplify: ans = ans + val; and its variants
// index = (index + 1) % n; // index++; if (index >= n) index = 0;
// index = (index + n - 1) % n; // index--; if (index < 0) index = n - 1;
// int ans = (int)((double)d + 0.5); // for rounding to nearest integer
// ans = min(ans, new_computation); // min/max shortcut
// alternative form but not used in this book: ans <?= new_computation;
// some code use short circuit && (AND) and || (OR)