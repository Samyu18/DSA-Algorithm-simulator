# 🎯 Algorithm Visualizer

A comprehensive web-based algorithm visualization tool built with Flask and modern JavaScript. Visualize and trace through various algorithms with step-by-step animations and detailed explanations.

## 🌟 Features

- **10 Algorithm Categories** with 30+ algorithms
- **Real-time Visualizations** with smooth animations
- **Step-by-step Tracing** with detailed explanations
- **Modern UI** with responsive design
- **Interactive Controls** for custom inputs
- **Multiple Visualization Types** (bars, tables, boards, graphs)

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   python app.py
   ```

3. **Open in Browser:**
   ```
   http://localhost:5001
   ```

## 📚 Algorithm Categories

### 1. Sorting Algorithms
- **Bubble Sort** - Simple comparison-based sorting
- **Selection Sort** - Find minimum and place at beginning
- **Insertion Sort** - Build sorted array one element at a time
- **Merge Sort** - Divide and conquer sorting
- **Quick Sort** - Partition-based sorting
- **Heap Sort** - Heap data structure-based sorting
- **Counting Sort** - Integer sorting with counting
- **Radix Sort** - Digit-by-digit sorting

### 2. Searching Algorithms
- **Linear Search** - Sequential search through array
- **Binary Search** - Divide and conquer search in sorted array
- **Jump Search** - Jump ahead and then linear search
- **Exponential Search** - Exponential jumps then binary search
- **Ternary Search** - Divide array into three parts

### 3. Greedy Algorithms
- **Activity Selection** - Select maximum non-overlapping activities
- **Fractional Knapsack** - Maximize value with fractional items

### 4. Dynamic Programming
- **0/1 Knapsack** - Binary choice knapsack problem
- **Longest Common Subsequence** - Find longest common subsequence
- **Fibonacci (DP)** - Fibonacci with memoization

### 5. Tree Algorithms
- **Tree Traversals** - Inorder, Preorder, Postorder traversals

### 6. Graph Algorithms
- **Breadth-First Search (BFS)** - Level-by-level traversal
- **Depth-First Search (DFS)** - Deep traversal
- **Dijkstra's Algorithm** - Shortest path algorithm

### 7. Backtracking
- **N-Queens Problem** - Place queens on chessboard

### 8. Bit Manipulation
- **Power of Two** - Check if number is power of 2
- **Count Set Bits** - Count number of 1s in binary

### 9. Sliding Window
- **Max Sum Subarray** - Maximum sum of k consecutive elements
- **Longest Substring No Repeat** - Longest substring without repeating characters

## 🎨 Visualization Types

### Array Visualizations
- **Bar Charts** for sorting and searching algorithms
- **Color-coded States**: Compare (yellow), Swap (red), Insert (green), Search (purple), Found (gold)

### Special Visualizations
- **Chess Board** for N-Queens problem
- **DP Tables** for dynamic programming algorithms
- **Graph Representations** for graph algorithms
- **Tree Structures** for tree algorithms

## 💡 How to Use

1. **Select Category**: Choose from the dropdown menu
2. **Pick Algorithm**: Select specific algorithm within the category
3. **Enter Input**: Fill in the required parameters
4. **Run & Visualize**: Click the button to see the algorithm in action
5. **Follow Trace**: Watch the step-by-step execution in the trace log

## 🔧 Input Formats

### Arrays
```
5,3,8,4,2,7,1,6
```

### Activities (for Activity Selection)
```
(1,4),(2,6),(3,5),(5,7),(6,8)
```

### Weights and Values (for Knapsack)
```
Weights: 2,3,4,5
Values: 3,4,5,6
Capacity: 10
```

### Strings (for LCS)
```
String 1: ABCDGH
String 2: AEDFHR
```

### Graph (for Graph Algorithms)
```json
{"A":["B","C"],"B":["A","D","E"],"C":["A","F"],"D":["B"],"E":["B","F"],"F":["C","E"]}
```

## 🛠️ Technical Details

### Backend (Flask)
- **Framework**: Flask with CORS support
- **Algorithm Implementation**: Pure Python with detailed tracing
- **API Endpoints**: RESTful API for algorithm execution
- **Error Handling**: Comprehensive error handling and validation

### Frontend (JavaScript)
- **Modern ES6+**: Async/await, arrow functions, destructuring
- **Responsive Design**: CSS Grid and Flexbox
- **Smooth Animations**: CSS transitions and transforms
- **Dynamic UI**: JavaScript-driven interface updates

### Styling
- **Modern CSS**: Gradients, shadows, rounded corners
- **Responsive**: Works on desktop and mobile
- **Accessibility**: Proper contrast and focus states
- **Animations**: Smooth transitions and loading states

## 📁 Project Structure

```
Sorting algorithms/
├── app.py              # Flask backend with all algorithms
├── index.html          # Frontend interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎯 Educational Value

This visualizer is perfect for:
- **Students** learning algorithms and data structures
- **Teachers** demonstrating algorithm concepts
- **Developers** understanding algorithm behavior
- **Interviews** practicing algorithm visualization

## 🔮 Future Enhancements

- [ ] More graph algorithms (Kruskal, Prim, Bellman-Ford)
- [ ] Advanced tree algorithms (AVL, Red-Black trees)
- [ ] String algorithms (KMP, Boyer-Moore)
- [ ] Network flow algorithms
- [ ] Machine learning algorithms
- [ ] Performance metrics and comparisons
- [ ] Export animations as GIF/MP4
- [ ] Mobile app version

## 🤝 Contributing

Feel free to contribute by:
- Adding new algorithms
- Improving visualizations
- Enhancing the UI/UX
- Fixing bugs
- Adding documentation

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Algorithm Learning! 🚀** # DSA-Algorithm-simulator
# DSA-Algorithm-simulator
# DSA-Algorithm-simulator
