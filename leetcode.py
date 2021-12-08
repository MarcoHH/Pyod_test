


def find_first(nums, target):
    l, r = 0, len(nums)-1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= target:
            r = mid

        else:
            l = mid+1
    return l if nums[l] == target else -1



def find_last(nums, target):
    l, r = 0, len(nums)-1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > target:
            r = mid - 1

        else:
            l = mid
    return l if nums[l] == target else -1

first = find_first([5,7,7,8,8,10], 8)
last = find_last([5,7,7,8,8,10], 8)

