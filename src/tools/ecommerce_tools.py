def check_stock(item_name: str) -> int:
    """Kiểm tra số lượng tồn kho của một sản phẩm."""
    item_name_lower = item_name.lower()
    if "iphone" in item_name_lower:
        return 50
    elif "macbook" in item_name_lower:
        return 10
    return 0

def get_discount(coupon_code: str) -> float:
    """Lấy số phần trăm giảm giá dựa vào mã giảm giá."""
    if coupon_code == "WINNER":
        return 0.10 # 10%
    elif coupon_code == "TET":
        return 0.20
    return 0.0

def calc_shipping(weight: float, destination: str) -> float:
    """Tính phí giao hàng dựa vào cân nặng (kg) và điểm đến."""
    dest_lower = destination.lower()
    base_cost = weight * 10000  # 10k per kg
    if "hanoi" in dest_lower or "hà nội" in dest_lower:
        return base_cost + 20000
    elif "hcm" in dest_lower or "hồ chí minh" in dest_lower:
        return base_cost + 30000
    return base_cost + 50000

def calc_total_price(price: float, quantity: int) -> str:
    """
    Tính tổng tiền dựa trên giá và số lượng.
    """
    total = price * quantity
    return f"Tổng tiền cho {quantity} sản phẩm là {total:,.0f} VND."

def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Lên mạng lấy tỷ giá và quy đổi tiền tệ qua API Internet."""
    import urllib.request
    import json
    try:
        from_c = from_currency.upper()
        to_c = to_currency.upper()
        url = f"https://api.exchangerate-api.com/v4/latest/{from_c}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            rates = data.get("rates", {})
            if to_c in rates:
                rate = rates[to_c]
                result = amount * rate
                return f"{amount:,.2f} {from_c} bằng khoảng {result:,.2f} {to_c} (Theo tỷ giá {rate} lấy trên mạng Internet)."
            else:
                return f"ERROR: Không hỗ trợ tỷ giá {to_c}"
    except Exception as e:
        return f"ERROR lỗi gọi API: {str(e)}"

# JSON Spec để đút cho Agent nhận diện
ECOMMERCE_TOOLS_SPEC = [
    {
        "name": "check_stock",
        "description": "Kiểm tra số lượng hàng tồn kho. Đầu vào nhận tham số `item_name` dạng string (ví dụ 'iPhone'). Nếu sản phẩm có tồn tại, trả về số lượng hiện có dạng nguyên (int). Nên dùng tool này đầu tiên để biết còn hàng không.",
        "parameters": "item_name: string"
    },
    {
        "name": "get_discount",
        "description": "Tính xem mã giảm giá ứng với bao nhiêu %. Đầu vào nhận tham số `coupon_code` dạng string (ví dụ 'WINNER'). Trả về tỉ lệ % dạng số thập phân float (ví dụ 0.1 là giảm 10%). Nếu trả về 0.0 nghĩa là mã hết hạng. Gọi tool này để trừ giá thành sản phẩm.",
        "parameters": "coupon_code: string"
    },
    {
        "name": "calc_shipping",
        "description": "Tính phí vận chuyển của đơn hàng. Đầu vào đòi 2 tham số: `weight` dạng float (cân nặng kiện hàng theo kg) và `destination` dạng string (Nơi đến như 'Hanoi'). Trả về tổng tiền phí giao hàng bằng VNĐ (float). Gọi tool này cộng vào bước cuối cùng.",
        "parameters": "weight: float, destination: string"
    },
    {
        "name": "calc_total_price",
        "description": "Tính tổng tiền sản phẩm. Đầu vào đòi 2 tham số: `price` dạng float (giá 1 sản phẩm) và `quantity` dạng int (số lượng sản phẩm). Trả về tổng tiền (vnd) của các sản phẩm đó.",
        "parameters": "price: float, quantity: int"
    },
    {
        "name": "convert_currency",
        "description": "Kết nối Internet API để tính tỷ giá ngoại tệ thật. Trả về kết quả sau quy đổi. Đầu vào: `amount` (số tiền dạng float), `from_currency` (Mã tiền gốc như 'USD', 'VND'), `to_currency` (Mã tiền đích như 'VND', 'EUR'). Thường dùng để quy tỷ giá tiền hàng.",
        "parameters": "amount: float, from_currency: string, to_currency: string"
    }
]
