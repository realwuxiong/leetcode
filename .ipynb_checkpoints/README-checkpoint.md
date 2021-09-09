# 力扣刷题
- [1、单调栈](#2020年12月18日)
- [2、动态规划](#2020年12月21日) 和[股票问题](#股票问题)
- [3、二叉树锯齿状层序遍历](#2020年12月12日)
- [4、移位](https://www.cnblogs.com/captainad/p/10968103.html)和[异或](#异或判断不同)
- [5、糖果问题（左右两次遍历取最大值）](#2020年12月24日)
- [6、贪心算法](#贪心算法汇总)
- [7、并查集，https://leetcode-cn.com/problems/evaluate-division/](#并查集)
- [8、滑动窗口](#滑动窗口) 
- [9、DFS/回溯算法](#回溯算法)
- [10、计算器（堆栈）](#计算器)
- [11、双指针](#双指针)
- [12、旋转数组](#旋转数组)
- [13、最低运载量（转化二分法）](#最低运载量（转化二分法）)
- [14、广度优先搜索](#广度优先搜索)
- [15、Boyer-Moore投票算法](#Boyer-Moore投票算法)
- [16、计数排序](#计数排序)
- [17、差分数组](#差分数组)
- [18、找到最终的安全状态(DFS+三色标记法)](#找到最终的安全状态(DFS+三色标记法))
- [19、最长回文子序列](#最长回文子序列)
- [20、用 Rand7() 实现 Rand10()](#用Rand7()实现Rand10())
- [21、螺旋矩阵](#螺旋矩阵)
- n皇后的题目搞一搞

## 螺旋矩阵
![](https://note.youdao.com/yws/api/personal/file/69ADBC35B4BA4E47AC3A8686ACEDE276?method=download&shareKey=aa5c0cc0495a0b1a0a78690376738660)

一共两种思路，都看看
59. 螺旋矩阵 II
```
# 螺旋矩阵
n=5
dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
matrix = [[0] * n for _ in range(n)]
row, col, dirIdx = 0, 0, 0
for i in range(n * n):
    matrix[row][col] = i + 1
    dx, dy = dirs[dirIdx]
    r, c = row + dx, col + dy
    if r < 0 or r >= n or c < 0 or c >= n or matrix[r][c] > 0:
        dirIdx = (dirIdx + 1) % 4   # 顺时针旋转至下一个方向
        dx, dy = dirs[dirIdx]
    row, col = row + dx, col + dy
print(matrix)

```

![](https://note.youdao.com/yws/api/personal/file/08838DCDFBC14D3AA564F2A9E5D948E3?method=download&shareKey=9a44cd3bb95fc7da32aec1d4f4f96ca3)


## 470. 用 Rand7()实现Rand10()
![](https://note.youdao.com/yws/api/personal/file/EE15EA52B96C4051AF0A223DF9C2FCAB?method=download&shareKey=7b0b783892cce98f5077bcfaed06bf9a)
```
# 470. 用 Rand7() 实现 Rand10()
# https://leetcode-cn.com/problems/implement-rand10-using-rand7/

# The rand7() API is already defined for you.
# def rand7():
# @return a random integer in the range 1 to 7

# 这个题目好好理解一下
class Solution:
    def rand10(self) -> int:
        while True:
            row = rand7()
            col = rand7()
            idx = (row - 1) * 7 + col
            if idx <= 40:
                return 1 + (idx - 1) % 10

```



## 最长回文子序列
![image.png](https://note.youdao.com/yws/api/personal/file/WEBd67287aea2668d7d13756cd2a9a8f8e7?method=download&shareKey=3a7dfeaf6cba56e0686020ff35633cf5)
```
# 516. 最长回文子序列
# https://leetcode-cn.com/problems/longest-palindromic-subsequence/

# 真的是躲不掉啊，回文字符迟早要遇到。
def longestPalindromeSubseq(s: str) -> int:
    # 采用动态规划？？
    n = len(s)
    dp = [[0]*n for _ in range(n)]

    #开始动态规划
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i+1,n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1]+2
            else:
                dp[i][j] = max(dp[i+1][j],dp[i][j-1])
    return dp[0][n-1]


    pass
s = "abaa"
longestPalindromeSubseq(s)
```

## 找到最终的安全状态(DFS+三色标记法)
![image.png](https://note.youdao.com/yws/res/8374/WEBRESOURCE61f39a5110a714646639ba2507648971)
```
# 802. 找到最终的安全状态
# https://leetcode-cn.com/problems/find-eventual-safe-states/

from typing import List
def eventualSafeNodes(graph: List[List[int]]) -> List[int]:
    """
    分析：有环则该节点为非安全节点，用dfs,
    还是看的答案，
    根据题意，若起始节点位于一个环内，或者能到达一个环，则该节点不是安全的。否则，该节点是安全的。

    我们可以使用深度优先搜索来找环，并在深度优先搜索时，用三种颜色对节点进行标记，标记的规则如下：

    白色（用 00 表示）：该节点尚未被访问；
    灰色（用 11 表示）：该节点位于递归栈中，或者在某个环上；
    黑色（用 22 表示）：该节点搜索完毕，是一个安全节点。
    当我们首次访问一个节点时，将其标记为灰色，并继续搜索与其相连的节点。

    如果在搜索过程中遇到了一个灰色节点，则说明找到了一个环，此时退出搜索，栈中的节点仍保持为灰色，这一做法可以将「找到了环」这一信息传递到栈中的所有节点上。

    如果搜索过程中没有遇到灰色节点，则说明没有遇到环，那么递归返回前，我们将其标记由灰色改为黑色，即表示它是一个安全的节点。

    """
    n = len(graph)
    visit = [0] * n
    ans = []
    def dfs(i):
        if visit[i] > 0:
            return  visit[i] == 2
        visit[i] = 1
        for j in graph[i]:
            if not dfs(j):
                return False
        visit[i] = 2
        return True
    return [i for i in range(n) if dfs(i)]
graph = [[1,2],[2,3],[5],[0],[5],[],[]]
eventualSafeNodes(graph)
```

## 差分数组
看这个题
1893. 检查是否区域内所有整数都被覆盖

看这个博客
https://blog.csdn.net/qq_31601743/article/details/105352885


## 计数排序
看这篇文章 https://www.cnblogs.com/xiaochuan94/p/11198610.html

和这个题目 https://leetcode-cn.com/problems/maximum-element-after-decreasing-and-rearranging/

## Boyer-Moore投票算法
面试题 17.10. 主要元素
https://leetcode-cn.com/problems/find-majority-element-lcci/solution/zhu-yao-yuan-su-by-leetcode-solution-xr1p/
 
这个解释不错
```
摩尔投票法，当年评论区大哥用打仗来做比喻的例子还很生动。

说多国开战，各方军队每次派一个士兵来两两单挑，每次单挑士兵见面一定会和对方同归于尽，最后只要哪边还有人活着就算胜利，那么最后一定是没有人活着，或者活下来的都是同一势力。

那么活下来的势力一定就是参战中势力最雄厚的嘛(指人最多)？不是的，假设总共有2n+1个士兵参战，其中n个属于一方，另n个属于另一方，最后一方势力只有一个人，也许前两方杀红了眼两败俱伤了，最后被剩下的一个人捡漏了也是可能的。

那么辛苦杀敌到底是为了什么呢？只为了两件事

最后活下来的势力未必就是人最多的(也许会被人偷鸡)
人最多的势力如果不能活下来，只说明它的势力还不够强大，不足以保证赢得战争的胜利(指人数超过总参战人数的一半)
如果最后没有人活下来，说明此次参战的势力中，没有任何一只足够强大到一定会赢得胜利。
严谨的逻辑证明咱不会。。凭理解写下思路，有错误的地方请大佬指正。

所以遍历一遍，每次清除一对不同势力的人，对最后活下来的势力单独验证一下究竟实力如何，对无人生还的情况，直接输出-1.
```

代码
```
# 20210709 力扣刷题
from typing import List
def majorityElement(nums: List[int]) -> int:
    candidate = -1
    count = 0
    for num in nums:
        if count == 0:
            candidate = num
        
        if num == candidate:
            count += 1
        else:
            count -= 1

    count = 0          
    for num in nums:
        if num == candidate:
            count += 1
            
    return candidate if count > len(nums)//2 else -1
nums = [3,3,4]
majorityElement(nums)
```







## 广度优先搜索
把思路好好捋一捋
752. 打开转盘锁
773. 滑动谜题
909. 蛇梯棋
815. 公交路线 
```
# 752. 打开转盘锁
# https://leetcode-cn.com/problems/open-the-lock/
# 我操？这一题完全没思路，明天做吧。。。。

# 20201年6月27日再做一次
# 广度搜索。做完用java写一遍
# 写的贼差，但是好歹算是写出了
from typing import List
from collections import deque
def openLock(deadends: List[str], target: str) -> int:
    statues = deque()
    statues.append(("0000",0))
    if "0000" in dead:
        return -1
    mySet = set()
    
    def enuStatue(s,count):
        ans = []
        s = list(s)
        for i in range(4):
            # 加一
            x = s[i]
            if s[i] == '9':
                s[i] = '0'
            else:
                s[i] = chr(ord(s[i])+1)
            if "".join(s) not in mySet and "".join(s) not in deadends:
                mySet.add("".join(s))
#                 print(ans,s)
                ans.append(("".join(s),count+1))
                statues.append(("".join(s),count+1))
            s[i] = x
            # 减一
            if s[i] == '0':
                s[i] = '9'
            else:
                s[i] = chr(ord(s[i])-1)
            if "".join(s) not in mySet and "".join(s) not in deadends:
                mySet.add("".join(s))
                ans.append(("".join(s),count+1))
                statues.append(("".join(s),count+1))
#             print(s,x,i)
            s[i] = x
#         print(ans)
        return ans
        
#     print(statues)
    while len(statues)>0:
        statue = statues.popleft()

        if statue[0] == target:
            return statue[1]
        enuStatue(statue[0],statue[1])

#         print(statues)
    return -1
        
        
    
    pass
deadends = ["0201","0101","0102","1212","2002"]
target = "0202"
openLock(deadends,target)
```



## 最低运载量/最大值最小问题（转化二分法）
```
# 1011. 在 D 天内送达包裹的能力
# https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/
from typing import List
def shipWithinDays(weights: List[int], D: int) -> int:
     # 确定二分查找左右边界
    left, right = max(weights), sum(weights)
    while left < right:
        mid = (left + right) // 2
        # need 为需要运送的天数
        # cur 为当前这一天已经运送的包裹重量之和
        need, cur = 1, 0
        for weight in weights:
            if cur + weight > mid:
                need += 1
                cur = 0
            cur += weight

        if need <= D:
            right = mid
        else:
            left = mid + 1

    return left

weights = [1,2,3,4,5,6,7,8,9,10]
D = 5
shipWithinDays(weights,D)
```
875. 爱吃香蕉的珂珂
```
# 二分法 确定最小速度，最小运载量
# 875. 爱吃香蕉的珂珂
# 还是用二分法试试
# 懂了方法就很easy
from typing import List
import math
def minEatingSpeed(piles: List[int], h: int) -> int:
    slow = 1
    fast = max(piles)
    while slow <= fast:
        mid = (slow+fast)//2
        need = 0
        for pile in piles:
            cur = math.ceil(pile/mid*1.0)
            need += cur
        if need > h:
            slow = mid + 1
        else:
            fast = mid -1
    return slow
    pass
piles = [3,6,7,11]
H = 8
minEatingSpeed(piles,H)
```
1231. 分享巧克力 (会员题) 
410. 分割数组的最大值
都是一个类型的题

## 旋转数组
看看力扣三个题

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

[81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

[154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

## 双指针
看看力扣这两题
26. 删除有序数组中的重复项
80. 删除有序数组中的重复项 II





## 计算器
力扣 150 224 227 1106四个题目看一看

```
# 四月的第一天，继续加油！
# 1006. 笨阶乘
# https://leetcode-cn.com/problems/clumsy-factorial/
# 10:04
import math
def clumsy(N: int) -> int:
    a = 0
    '''
    0:*
    1:/
    2:+
    3:-
    '''
    # 可以不用递归，也可以用递归，先不用递归试试
    # 用栈，哎，做了几次这样的题居然还没做出来
    stack = []
    stack.append(N)
    N -= 1
    while N > 0:
        if a % 4==0:
            stack.append(stack.pop()*N)
        elif a %4 == 1:
            tem = stack.pop()/N
            if tem >= 0:
                stack.append(math.floor(tem))
            else:
                stack.append(math.ceil(tem)) 
        elif a % 4 == 2:
            stack.append(N)
        else:
            stack.append(-N)
        N -= 1
        a += 1
    print(sum(stack))
            
clumsy(10)
```

根据求解问题「150. 逆波兰表达式求值」、「224. 基本计算器」、「227. 基本计算器 II」的经验，表达式的计算一般可以借助数据结构「栈」完成，特别是带有括号的表达式。我们将暂时还不能确定的数据存入栈，确定了优先级最高以后，一旦可以计算出结果，我们就把数据从栈里取出，整个过程恰好符合了「后进先出」的规律。本题也不例外。

根据题意，「笨阶乘」没有显式括号，运算优先级是先「乘除」后「加减」。我们可以从 NN 开始，枚举 N - 1N−1、N-2N−2 直到 11 ，枚举这些数的时候，认为它们之前的操作符按照「乘」「除」「加」「减」交替进行。

出现乘法、除法的时候可以把栈顶元素取出，与当前的 NN 进行乘法运算、除法运算（除法运算需要注意先后顺序），并将运算结果重新压入栈中；

出现加法、减法的时候，把减法视为加上一个数的相反数，然后压入栈，等待以后遇见「乘」「除」法的时候取出。

最后将栈中元素累加即为答案。由于加法运算交换律成立，可以将栈里的元素依次出栈相加。






## 回溯算法
https://leetcode-cn.com/problems/generate-parentheses/

数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

```
from typing import List
def generateParenthesis(n: int) -> List[str]:
        res = []
        S = []
        def dfs(left,right,S):
            if len(S) == 2*n:
               res.append(''.join(S))
               return
            if left < n:
                S.append("(")
                dfs(left+1,right,S)
                S.pop()
            if left > right:
                S.append(")")
                dfs(left,right+1,S)
                S.pop()
        dfs(0,0,S)
        return res
generateParenthesis(3)
```

[力扣第78题](https://leetcode-cn.com/problems/subsets/solution/)

[力扣第46题](https://leetcode-cn.com/problems/permutations/)




## 滑动窗口
把这个题目解析后面的所有题目搞清楚
https://leetcode-cn.com/problems/subarrays-with-k-different-integers/solution/k-ge-bu-tong-zheng-shu-de-zi-shu-zu-by-l-ud34/

这个题目好好看一看
https://leetcode-cn.com/classic/problems/longest-repeating-character-replacement/description/


## 并查集
1、[省份问题](https://leetcode-cn.com/problems/number-of-provinces/)

```
# 2021年1月19日 好好捋一捋并查集

# 第一题  省份数量

def findCircleNum(isConnected) -> int:
    # fuck 并查集
    def find(i):
        if(parents[i]!=i):
            parents[i]=find(parents[i])
        return parents[i]
    def union(i,j):
        parents[find(i)]=find(j)
        pass
    parents = [i for i in range(len(isConnected))]
    
    for i in range(len(isConnected)):
        for j in range(i+1,len(isConnected[0])):
            if isConnected[i][j]==1:
                union(i,j)
    sumProvince  = 0 
    print(parents)
    for i,data in enumerate(parents):
        if i == data:
            sumProvince += 1
    return sumProvince
    
isConnected = [[1,1,0],[1,1,0],[0,0,1]]
findCircleNum(isConnected)

```
i 和 j 联通 那个j就是i的parent，然后find其实是一个递归函数。

2、[200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```
# 200. 岛屿数量
# https://leetcode-cn.com/problems/number-of-islands/
from typing import List
def numIslands(grid: List[List[str]]) -> int:
    """
    这个题目是二维的并查集,看的别人的答案
    """
    f = {}
    def find(x):
        f.setdefault(x,x)
        if f[x]!=x:
            f[x] = find(f[x])
        return f[x]
    def union(x,y):
        f[find(y)] = find(x)

    if not grid:
        return 0
    row,col =len(grid),len(grid[0])
    for i in range(row):
        for j in range(col):
            if grid[i][j] == "1":
                for x, y in [[-1, 0], [0, -1]]:
                    tmp_i = i + x
                    tmp_j = j + y
                    if 0 <= tmp_i < row and 0 <= tmp_j < col and grid[tmp_i][tmp_j] == "1":
                        union(tmp_i * col + tmp_j, i * col + j)
    res = set()
    for i in range(row):
        for j in range(col):
            if grid[i][j] == "1":
                res.add(find(col*i+j))
    return len(res)
    pass
grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]

numIslands(grid)


```
3、130.被围绕的区域
```
# 130.被围绕的区域,看的别人的答案
# https://leetcode-cn.com/problems/surrounded-regions/
from typing import List
def solve(board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    f = {}
    def find(x):
        f.setdefault(x,x)
        if f[x]!=x:
            f[x] = find(f[x])
        return f[x]
    def union(x,y):
        f[find(y)] = find(x)
    if not board or not board[0]:
        return
    row,col = len(board),len(board[0])
    dummy = row*col
    for i in range(row):
        for j in range(col):
            if board[i][j] == "O":
                if i == 0 or i == row - 1 or j == 0 or j == col - 1:
                    union(i * col + j, dummy)
                else:
                    for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if board[i + x][j + y] == "O":
                            union(i * col + j, (i + x) * col + (j + y))

    for i in range(row):
        for j in range(col):
            if find(dummy) == find(i * col + j):
                board[i][j] = "O"
            else:
                board[i][j] = "X"
    print(board)
board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
solve(board)

```

## 股票问题
https://leetcode-cn.com/circle/article/qiAgHn/

## 贪心算法汇总
&nbsp;&nbsp;&nbsp;&nbsp;贪心算法（又称贪婪算法）是指，在对问题求解时，总是做出在当前看来是最好的选择。也就是说，不从整体最优上加以考虑，他所做出的是在某种意义上的局部最优解。

&nbsp;&nbsp;&nbsp;&nbsp;贪心算法不是对所有问题都能得到整体最优解，关键是贪心策略的选择，选择的贪心策略必须具备无后效性，即某个状态以前的过程不会影响以后的状态，只与当前状态有关。

[力扣贪心算法的题目](https://leetcode-cn.com/tag/greedy/)

例子一

[最大的团队表现值](https://leetcode-cn.com/problems/maximum-performance-of-a-team/)

小根堆。。。堆排序。。这个思想要学一学

```
class Solution:
    class Staff:
        def __init__(self, s, e):
            self.s = s
            self.e = e
        
        def __lt__(self, that):
            return self.s < that.s
        
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        v = list()
        for i in range(n):
            v.append(Solution.Staff(speed[i], efficiency[i]))
        v.sort(key=lambda x: -x.e)

        q = list()
        ans, total = 0, 0
        for i in range(n):
            minE, totalS = v[i].e, total + v[i].s
            ans = max(ans, minE * totalS)
            heapq.heappush(q, v[i])
            total += v[i].s
            if len(q) == k:
                item = heapq.heappop(q)
                total -= item.s
        return ans % (10**9 + 7)

```


## 2020年12月24日
今天做的题目是

[135. 分发糖果](https://leetcode-cn.com/problems/candy/)
![](https://note.youdao.com/yws/api/personal/file/08E915CD33AB44589C63EEE1F2223D71?method=download&shareKey=d389546a8f0151feeca056b8b8f29758)

主要的思想就是从左和从右满足要求，然后取最大值。

```

class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        left = [0] * n
        for i in range(n):
            if i > 0 and ratings[i] > ratings[i - 1]:
                left[i] = left[i - 1] + 1
            else:
                left[i] = 1
        
        right = ret = 0
        for i in range(n - 1, -1, -1):
            if i < n - 1 and ratings[i] > ratings[i + 1]:
                right += 1
            else:
                right = 1
            ret += max(left[i], right)
        
        return ret
```





## 2020年12月22日
今天做的题目是

[103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)
![](https://note.youdao.com/yws/api/personal/file/1A07B5423CA7478C9D0AAF595E97A303?method=download&shareKey=f10776912e8cd4998d0b54c40c5539e2)

思路：
1. 就是层次遍历，然后加了一个flag，从头或者从尾巴插入数字。
2. while里面的for循环很巧妙。
3. 代码参考的力扣官方的代码，我用的是python的改写。
4. 可以看看这个[博客](https://www.cnblogs.com/ACStrive/p/11222390.html)


```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        # 层次遍历是需要队列的。
        queue=[]
        res=[]
        if root is None:
            return []
        p = root
        queue.append(p)
        flag = False
        while queue:
            layer=[]
            size = len(queue)

            for _ in range(size):
                temp = queue.pop(0)
                if flag:
                    layer.insert(0,temp.val)
                else:
                    layer.append(temp.val)
                if temp.left!=None:
                    queue.append(temp.left)
                if temp.right!=None:
                    queue.append(temp.right)
            flag = bool(1-flag)  
            res.append(layer)
 
        return res
```



## 2020年12月21日
---
学习学习动态规划，动态规划方程。理解动态规划思想。

[这个博客讲的很不错](https://blog.csdn.net/zw6161080123/article/details/80639932)

[01背包问题](https://www.cnblogs.com/kkbill/p/12081172.html)

[使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)
![](https://note.youdao.com/yws/api/personal/file/06ACE5AF2E7540DFAF76A394CE4068CB?method=download&shareKey=12960c897cede0d4acf7f46dc70579b7)

```
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        prev = curr = 0
        for i in range(2, n + 1):
            nxt = min(curr + cost[i - 1], prev + cost[i - 2])
            prev, curr = curr, nxt
        return curr

```

115. 不同的子序列，力扣刷题-20210317 可以采用回溯，但是超时了
https://leetcode-cn.com/problems/distinct-subsequences/submissions/
![](https://note.youdao.com/yws/api/personal/file/725ABC447949414F80674E9BD209154A?method=download&shareKey=174915c6ecf924e5308073e3b80f1749)

最长回文字符串
https://leetcode-cn.com/problems/longest-palindromic-substring/

39. 组合总和

377. 组合总和 Ⅳ
https://leetcode-cn.com/problems/combination-sum-iv/
```
# 377. 组合总和 Ⅳ
# https://leetcode-cn.com/problems/combination-sum-iv/
# dfs,啪一下就出来了，但是超时了
from typing import List
def combinationSum4(nums: List[int], target: int) -> int:
    n = len(nums)
    res=[]
    stack=[]
    def dfs(i):
        if i > len(nums) or sum(stack)>target:
            return
        if sum(stack) == target:
            res.append(stack[:])
        for j in range(n):
            stack.append(nums[j])
            dfs(j+1)
            stack.pop()
    dfs(0)
    print(res)

nums = [4,2,1]
target = 4
combinationSum4(nums,target)

# 方法二，动态规划
from typing import List
def combinationSum4(nums: List[int], target: int) -> int:
    n = len(nums)
    dp = [1]+[0]*target
    for i in range(1,target+1):
        for num in nums:
            if num <= i:
                dp[i] += dp[i - num]
        print(dp)
    print(dp[target])
nums = [4,2,1]
target = 4
combinationSum4(nums,target)
```

## 2020年12月18日
----
移位，异或，单调栈

单调栈

https://leetcode-cn.com/problems/trapping-rain-water/

https://leetcode-cn.com/problems/daily-temperatures/

https://leetcode-cn.com/problems/largest-rectangle-in-histogram/

![](https://note.youdao.com/yws/api/personal/file/8AE26029D31D4D5AAEB67AF0E429DE2A?method=download&shareKey=ecb4089b714df136ef333f662ec19824)


```
# 每日温度
# https://leetcode-cn.com/problems/daily-temperatures/
temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
def Solution(temperatures):
    output=[0]*len(temperatures)
    #单调栈
    stack = []
    for index in range(len(temperatures)):
        while(stack and temperatures[index] > temperatures[stack[-1]]):
            output[stack[-1]] = index - stack[-1]
            stack.pop()
        stack.append(index)
    return output
print(Solution(temperatures))

```

![](https://note.youdao.com/yws/api/personal/file/2DF6F9106468462A85821A377DB57E74?method=download&shareKey=4fd4b4e021ca340ee52cbb4a4de8e652)

```
# https://leetcode-cn.com/problems/trapping-rain-water/
# 接雨水

height = [0,1,0,2,1,0,1,3,2,1,2,1]
def rainWater(height):
    ans = 0
    stack =[]
    for current in range(len(height)):
        while(len(stack) >0 and height[current] > height[stack[-1]]):
            top = stack.pop()
            if len(stack) == 0:
                break
            distance = current - stack[-1] -1
            ans = ans+ distance * ( min(height[stack[-1]],height[current]) -height[top])
            
        stack.append(current)
    return ans

print(rainWater(height))
```

## 异或判断不同

https://leetcode-cn.com/problems/single-number/solution/zhi-chu-xian-yi-ci-de-shu-zi-by-leetcode-solution/

![](https://note.youdao.com/yws/api/personal/file/13E0D6E2D55944978F757F4377056583?method=download&shareKey=da7e715a01f95711e97616236cefcd30)

异或有交换律和结合律

https://leetcode-cn.com/problems/counting-bits/solution/bi-te-wei-ji-shu-by-leetcode-solution-0t1i/
这是移位的一个题