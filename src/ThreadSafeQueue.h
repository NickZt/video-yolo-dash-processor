#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

template <typename T> class ThreadSafeQueue {
public:
  ThreadSafeQueue(size_t maxSize = 100) : maxSize_(maxSize), closed_(false) {}

  ~ThreadSafeQueue() { close(); }

  bool push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    condVarPush_.wait(lock,
                      [this]() { return queue_.size() < maxSize_ || closed_; });

    if (closed_)
      return false;

    queue_.push(std::move(item));
    condVarPop_.notify_one();
    return true;
  }

  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    condVarPop_.wait(lock, [this]() { return !queue_.empty() || closed_; });

    if (queue_.empty() && closed_) {
      return std::nullopt;
    }

    T item = std::move(queue_.front());
    queue_.pop();
    condVarPush_.notify_one();
    return std::move(item);
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    condVarPush_.notify_all();
    condVarPop_.notify_all();
  }

  bool is_closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_ && queue_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable condVarPush_;
  std::condition_variable condVarPop_;
  size_t maxSize_;
  bool closed_;
};
