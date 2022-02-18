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
